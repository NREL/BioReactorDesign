/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2020 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kLA.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(kLA, 0);
    addToRunTimeSelectionTable(functionObject, kLA, dictionary);
}
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::kLA::kLA
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    logFiles(obr_,name),
    phases_(mesh_.lookupObject<phaseSystem>("phaseProperties").phases()),
    continuumPhase_(dict.lookup<word>("continuumPhase")),
    dispersedPhase_(dict.lookup<word>("dispersedPhase")),
    species_(dict.lookup<word>("species")),
    phaseMin_(dict.lookup<scalar>("phaseMin")),
    interfaceMax_(dict.lookup<scalar>("interfaceMax")),
    data_(4*species_.size())
{

    functionObject::read(dict);
    resetName(typeName);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::functionObjects::kLA::~kLA()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::functionObjects::kLA::writeFileHeader(const label i)
{
    if (Pstream::master())
    {
        writeCommented(file(), "time");

        forAll(species_, specieI)
        {
            word speciesName = species_[specieI];
            writeTabbed(file(), speciesName );
            writeTabbed(file(), speciesName + "_star");
            writeTabbed(file(), speciesName + "_mdot");
            writeTabbed(file(), speciesName + "_kLA");
        }

        file() << endl;
    }
}

bool Foam::functionObjects::kLA::execute()
{
    const volScalarField& alphac = mesh_.lookupObject<volScalarField>("alpha." + continuumPhase_);
    const scalarField& V = mesh().V();
    const volScalarField& rho = mesh_.lookupObject<volScalarField>("rho." + continuumPhase_);

    autoPtr<volScalarField> interface;

    if (mesh_.foundObject<volScalarField>("interface." + dispersedPhase_ + continuumPhase_))
    {
        interface.set( &const_cast<volScalarField&>(mesh_.lookupObject<volScalarField>("interface." + dispersedPhase_ + continuumPhase_)) );
    }

    data_ *= 0.;

    scalar totMassLiq {0.};
    scalar totVolLiq {0.};
    forAll(V, cellI)
    {
        if ( alphac[cellI] < phaseMin_ ) continue;
        if ( !interface.empty() ) { if (interface()[cellI] > interfaceMax_) continue; };

        const scalar liqVol = alphac[cellI]*V[cellI];
        const scalar liqMass = liqVol*rho[cellI];
        totMassLiq += liqMass;
        totVolLiq += liqVol;
    }

    reduce(totMassLiq, sumOp<scalar>());
    reduce(totVolLiq, sumOp<scalar>());


    forAll(species_, specieI)
    {
        const label inId {specieI*4};
        const word& specieName = species_[specieI];

        const volScalarField& cl = mesh_.lookupObject<volScalarField>(specieName + "." + continuumPhase_);
        const volScalarField::Internal& SP = mesh_.lookupObject<volScalarField::Internal>("phaseChange:mDot" + specieName + "Sp");
        const volScalarField::Internal& SU = mesh_.lookupObject<volScalarField::Internal>("phaseChange:mDot" + specieName + "Su"); 

        forAll(SP, cellI)
        {
            if ( alphac[cellI] < phaseMin_ ) continue;
            if ( !interface.empty() ) { if (interface()[cellI] > interfaceMax_) continue; };

            const scalar liqVol = alphac[cellI]*V[cellI]/(totVolLiq+1e-32);
            const scalar liqMass = liqVol*rho[cellI]/(totMassLiq+1e-32);
            data_[inId] += cl[cellI] * liqMass;
            data_[inId + 1] += liqMass * SU[cellI] / ( SP[cellI] + 1e-32 );
            data_[inId + 2] += liqVol * ( SU[cellI] + ( SP[cellI] * cl[cellI] ) );
//            data_[inId + 3] += liqVol *  SP[cellI] ;
        }
    }

    reduce(data_,sumOp<scalarField>());

    forAll(species_, specieI)
    {
        data_[specieI*4 + 3] = data_[specieI*4 + 2] / ( data_[specieI*4 + 1] + data_[specieI*4] + 1e-32);
    }

    return true;
}


bool Foam::functionObjects::kLA::write()
{
    Info << logFiles::names();
    logFiles::write();

    if (Pstream::master())
    {
            file() << mesh_.time().name();
            forAll(data_,dI)
            {
                file() << tab;
                file() << data_[dI];
            }
            file() << endl;
    }

    return true;
}


// ************************************************************************* //
