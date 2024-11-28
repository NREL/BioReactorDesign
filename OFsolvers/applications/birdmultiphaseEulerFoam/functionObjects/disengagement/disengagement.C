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

#include "disengagement.H"
#include "addToRunTimeSelectionTable.H"
#include "fvCFD.H"
#include "fixedValueFvPatchField.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(disengagement, 0);
    addToRunTimeSelectionTable(functionObject, disengagement, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::disengagement::disengagement
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    logFiles(obr_,name),
    phases_(mesh_.lookupObject<phaseSystem>("phaseProperties").phases()),
    phaseName_(dict.lookup<word>("phase")),
    inletPatch_(dict.lookup<word>("inlet")),
    holdup_(),
    tolerance_(dict.lookup<scalar>("tolerance")),
    nsamples_(dict.lookup<label>("nsamples")),
    disengaged_(false),
    writtenAt_(0)
{
    //- Check that U has fixedValue
    volVectorField& U = const_cast<volVectorField&>(mesh_.lookupObject<volVectorField>("U." + phaseName_));
    label patchI = mesh_.boundaryMesh().findPatchID(inletPatch_);
    volVectorField::Boundary& UBf = const_cast<volVectorField::Boundary&>(U.boundaryFieldRef());

    if (!isA<fixedValueFvPatchVectorField>(UBf[patchI]))
    {
        FatalIOErrorInFunction(dict)
            << "Incorrect boundary condition for U." << phaseName_ << " at patch " << inletPatch_ << "\n"
            << "You must use fixedValue with this functionObject."
            << exit(FatalIOError);
    }

    functionObject::read(dict);
    resetName(typeName);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::functionObjects::disengagement::~disengagement()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::functionObjects::disengagement::writeFileHeader(const label i)
{
    if (Pstream::master())
    {
        writeHeader(file(), "holdup");
        writeCommented(file(), "time");
        writeTabbed(file(), "holdup");

        file() << endl;
    }
}

bool Foam::functionObjects::disengagement::execute()
{
    const volScalarField& alpha = mesh_.lookupObject<volScalarField>("alpha." + phaseName_);
                    
    //- Compute holdup
    scalar volume = gSum(mesh_.V());
    scalar holdup = gSum(fvc::volumeIntegrate(alpha))/volume;
    Pair<scalar> holddata(mesh_.time().value(), holdup);

    holdup_.append(holddata);

    //- Stop if already disengaged
    if(disengaged_) return true;

    //- Check if the average over the last  samples
    if (holdup_.size() > 2*nsamples_ + 1)
    {
        scalar holdup_mean_long(0.);
        scalar holdup_mean_short(0.);
        scalar t0_long(holdup_[holdup_.size() - 2*nsamples_].first());
        scalar deltat_long(0.);
        scalar deltat_short(0.);

        t0_long = holdup_[holdup_.size() - 2*nsamples_ -1].first();


        for (int i = holdup_.size() - 2*nsamples_; i < holdup_.size(); i++)
        {
            scalar dt = holdup_[i].first() - t0_long;
            t0_long = holdup_[i].first();
            holdup_mean_long += dt*holdup_[i].second();
            deltat_long += dt;

            if (i >= holdup_.size() - nsamples_)
            {
                holdup_mean_short +=  dt*holdup_[i].second();
                deltat_short += dt;
            }
            
        }

        holdup_mean_long /= deltat_long;
        holdup_mean_short /= deltat_short;

        if (mag(holdup_mean_long - holdup_mean_short) < tolerance_)
        {
            if(!disengaged_)
            {
                Info << "functionObject::disengagement: Disengaging!\n";
            }

            disengaged_ = true;

            //- Get boundary condition
            volVectorField& U = const_cast<volVectorField&>(mesh_.lookupObject<volVectorField>("U." + phaseName_));

            label patchI = mesh_.boundaryMesh().findPatchID(inletPatch_);

            volVectorField::Boundary& UBf = const_cast<volVectorField::Boundary&>(U.boundaryFieldRef());

            //- Stop the flow from entering
            UBf[patchI] = vector(0,0,0);
        
        }
                   
    }
                
    return true;
}


bool Foam::functionObjects::disengagement::write()
{
    Info << logFiles::names();
    logFiles::write();

    if (Pstream::master())
    {
        for (label i = writtenAt_; i < holdup_.size(); i++)
        {
            file() << holdup_[i].first();
            file() << tab;
            file() << holdup_[i].second();
            file() << endl;
        }


    }

    writtenAt_ = holdup_.size() - 1;

    return true;
}


// ************************************************************************* //
