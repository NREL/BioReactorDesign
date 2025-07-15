/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2015-2023 OpenFOAM Foundation
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

#include "BioReactingPhaseModel.H"
#include "phaseSystem.H"
#include "fvMatrix.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasePhaseModel>
Foam::BioReactingPhaseModel<BasePhaseModel>::BioReactingPhaseModel
(
    const phaseSystem& fluid,
    const word& phaseName,
    const bool referencePhase,
    const label index
)
:
    BasePhaseModel(fluid, phaseName, referencePhase, index),
    dict_(fluid.subDict("bioReactions")),
    species_(dict_.lookup("species")),
    maxUptakeRate_(),
    K_(),
    URSp_()
    // DCW_
    // (
    //     IOobject
    //     (
    //         "DCW",
    //         this->mesh().time().name(),
    //         this->mesh(),
    //         IOobject::MUST_READ,
    //         IOobject::AUTO_WRITE
    //     ),
    //     this->mesh()
    // )
{
    forAll(species_, specieI)
    {
        dictionary subsp(dict_.subDict(species_[specieI]));

        maxUptakeRate_.append(subsp.lookup("maxUptakeRate"));
        K_.append(subsp.lookup("K"));
        
        URSp_.set(
            specieI,
            new volScalarField
            (
                IOobject
                (
                    phaseName + ":URSp" + species_[specieI].capitalise(),
                    this->mesh().time().name(),
                    this->mesh(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                this->mesh(),
                dimensionedScalar(dimDensity/dimTime, 0)
            )
        );
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class BasePhaseModel>
Foam::BioReactingPhaseModel<BasePhaseModel>::~BioReactingPhaseModel()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasePhaseModel>
void Foam::BioReactingPhaseModel<BasePhaseModel>::correctReactions()
{

    forAll(this->Y(), speciei)
    {
        forAll(species_, speciej)
        {
            if ( this->Y()[speciei].name() == IOobject::groupName(species_[speciej], this->name()) )
            {
               volScalarField& URSp = URSp_[speciej];  
               URSp =  - maxUptakeRate_[speciej] / (  K_[speciej] + this->Y()[speciei] );
            }
        }
    }

    BasePhaseModel::correctReactions();
}


template<class BasePhaseModel>
Foam::tmp<Foam::volScalarField::Internal>
Foam::BioReactingPhaseModel<BasePhaseModel>::R(const label speciei) const
{
    forAll(species_, speciej)
    {
        if ( this->Y()[speciei].name() == IOobject::groupName(species_[speciej], this->name()) )
        {
            return URSp_[speciej];
        }
    }

    return
        volScalarField::Internal::New
        (
            IOobject::groupName("R_" + this->Y()[speciei].name(), this->name()),
            this->mesh(),
            dimensionedScalar(dimDensity/dimTime, 0)
        );
}


template<class BasePhaseModel>
Foam::tmp<Foam::fvScalarMatrix> Foam::BioReactingPhaseModel<BasePhaseModel>::R
(
    volScalarField& Yi
) const
{
    forAll(species_, speciej)
    { 
        if ( Yi.name() == IOobject::groupName(species_[speciej], this->name()) )
        {            
            return fvm::Sp(URSp_[speciej],Yi);
        }
    }

    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix(Yi, dimMass/dimTime)
    );
}

template<class BasePhaseModel>
Foam::tmp<Foam::volScalarField>
Foam::BioReactingPhaseModel<BasePhaseModel>::Qdot() const
{
    return volScalarField::New
    (
        IOobject::groupName("Qdot", this->name()),
        this->mesh(),
        dimensionedScalar(dimEnergy/dimTime/dimVolume, 0)
    );
}


// ************************************************************************* //
