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
    OURmax_(dict_.lookup("OURmax")),
    K_(dict_.lookup("K")),
    DCW_
    (
        IOobject
        (
            "DCW",
            this->mesh().time().name(),
            this->mesh(),
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh()
    )
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class BasePhaseModel>
Foam::BioReactingPhaseModel<BasePhaseModel>::~BioReactingPhaseModel()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasePhaseModel>
void Foam::BioReactingPhaseModel<BasePhaseModel>::correctReactions()
{
    BasePhaseModel::correctReactions();
}


template<class BasePhaseModel>
Foam::tmp<Foam::volScalarField::Internal>
Foam::BioReactingPhaseModel<BasePhaseModel>::R(const label speciei) const
{
    if ( this->Y()[speciei].name() == IOobject::groupName(species_, this->name()) )
    {
        volScalarField coeff( -OURmax_ / ( ( this->Y()[speciei]*K_ / this->fluid().rho() )  + this->Y()[speciei] ) );
        return coeff.internalField();
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
    if ( Yi.name() == IOobject::groupName(species_, this->name()) )
    {
        volScalarField coeff( OURmax_ / ( ( K_ / this->fluid().rho() )  + Yi ) );
        return -fvm::Sp( coeff, Yi );
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
