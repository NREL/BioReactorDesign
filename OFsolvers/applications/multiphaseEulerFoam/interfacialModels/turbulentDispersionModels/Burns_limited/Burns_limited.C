/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2014-2021 OpenFOAM Foundation
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

#include "Burns_limited.H"
#include "phasePair.H"
#include "phaseDynamicMomentumTransportModel.H"
#include "addToRunTimeSelectionTable.H"

#include "dragModel.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace turbulentDispersionModels
{
    defineTypeNameAndDebug(Burns_limited, 0);
    addToRunTimeSelectionTable
    (
        turbulentDispersionModel,
        Burns_limited,
        dictionary
    );
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::turbulentDispersionModels::Burns_limited::Burns_limited
(
    const dictionary& dict,
    const phasePair& pair
)
:
    turbulentDispersionModel(dict, pair),
    sigma_("sigma", dimless, dict),
    height_lim_("height_lim", dimless, dict),
    height_dir_("height_dir", dimless, dict)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::turbulentDispersionModels::Burns_limited::~Burns_limited()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::turbulentDispersionModels::Burns_limited::D() const
{
    const fvMesh& mesh(pair_.phase1().mesh());
    const dragModel& drag =
        mesh.lookupObject<dragModel>
        (
            IOobject::groupName(dragModel::typeName, pair_.name())
        );

    Foam::tmp<Foam::volScalarField> D = drag.Ki()
       *continuousTurbulence().nut()
       /sigma_
       *pair_.dispersed()
       *sqr(pair_.dispersed() + pair_.continuous())
       /(
            max(pair_.dispersed(), pair_.dispersed().residualAlpha())
           *max(pair_.continuous(), pair_.continuous().residualAlpha())
        );
 
    volVectorField centres = D.ref().mesh().C();
    int dir = int (height_dir_.value());
    forAll(centres, cellI) {
        if (centres[cellI][dir]>height_lim_.value()) {
            D.ref()[cellI] = 0.0;
        }
    }

    return D;
}


// ************************************************************************* //
