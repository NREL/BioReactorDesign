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

//- Add an element to the back of a list and shift all elements left.
//  Removes first element. This allows to only keep the information needed.
template<class Type>
void add_to_list_like_static_queue(List<Type>& list, const Type& value)
{
    for (label i = 0; i < list.size() - 1; i++)
    {
        //- Movel all elements
        list[i] = list[i+1];
    }

    //- emplace last element
    list[list.size()-1] = value;
    
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
    inletPhaseName_(dict.lookup<word>("inletPhase")),
    inletPatch_(dict.lookup<word>("inlet")),
    tolerance_(dict.lookup<scalar>("tolerance")),
    nsamples_(dict.lookup<label>("nsamples")),
    direction_(dict.lookup<vector>("direction")),
    disengage_(dict.lookup<bool>("disengage")),
    phase_com_(2*nsamples_,{0.,-1.}),
    disengaged_(false)
{
    //- Check that U has fixedValue
    volVectorField& U = const_cast<volVectorField&>(mesh_.lookupObject<volVectorField>("U." + inletPhaseName_));
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
        writeHeader(file(), "phase_com");
        writeCommented(file(), "time");
        writeTabbed(file(), "phase_com");
        writeTabbed(file(), "disengaged");

        file() << endl;
    }
}

bool Foam::functionObjects::disengagement::execute()
{
    const volScalarField& alpha = mesh_.lookupObject<volScalarField>("alpha." + phaseName_);
                    
    //- Compute phase_com
    scalar volume = gSum(fvc::volumeIntegrate(alpha));
    volScalarField hcoord = mesh_.C()&direction_;
    scalar phase_com = gSum(fvc::volumeIntegrate(hcoord*alpha))/volume;
    Pair<scalar> holddata(mesh_.time().value(), phase_com);

    //- See function at the beginning of file
    add_to_list_like_static_queue(phase_com_,holddata);

    //- Skip rest if no disengagement is required
    if (!disengage_) return true;

    //- Stop if already disengaged
    if (disengaged_ ) return true;

    //- Do the check only if the list is complete
    if (phase_com_[0].second() > 0.)
    {
        scalar phase_com_mean_long(0.);
        scalar phase_com_mean_short(0.);
        scalar t0_long(phase_com_[0].first());
        scalar deltat_long(0.);
        scalar deltat_short(0.);

        //- The first (oldest) sample is skipped to have a well-defined dt
        for (int i = 1; i < 2*nsamples_; i++)
        {
            scalar dt = phase_com_[i].first() - t0_long;
            t0_long = phase_com_[i].first();
            phase_com_mean_long += dt*phase_com_[i].second();
            deltat_long += dt;

            //- Compute average on the most recent samples
            if (i >= nsamples_)
            {
                phase_com_mean_short +=  dt*phase_com_[i].second();
                deltat_short += dt;
            }
            
        }

        phase_com_mean_long /= deltat_long;
        phase_com_mean_short /= deltat_short;

        if (mag(phase_com_mean_long - phase_com_mean_short) < tolerance_)
        {
            if(!disengaged_)
            {
                Info << "functionObject::disengagement: Disengaging!\n";
            }

            disengaged_ = true;

            //- Get boundary condition
            volVectorField& U = const_cast<volVectorField&>(mesh_.lookupObject<volVectorField>("U." + inletPhaseName_));

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

    label dis = 0;
    if(disengaged_) dis = 1;

    if (Pstream::master())
    {
            file() << phase_com_[2*nsamples_ -1].first();
            file() << tab;
            file() << phase_com_[2*nsamples_ -1].second();
            file() << tab;
            file() << dis;
            file() << endl;
     
    }

    return true;
}


// ************************************************************************* //
