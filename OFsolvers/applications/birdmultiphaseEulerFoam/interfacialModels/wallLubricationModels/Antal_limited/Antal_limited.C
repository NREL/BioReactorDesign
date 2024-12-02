/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2014-2020 OpenFOAM Foundation
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

#include "Antal_limited.H"
#include "phasePair.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace wallLubricationModels
{
    defineTypeNameAndDebug(Antal_limited, 0);
    addToRunTimeSelectionTable
    (
        wallLubricationModel,
        Antal_limited,
        dictionary
    );
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::wallLubricationModels::Antal_limited::Antal_limited
(
    const dictionary& dict,
    const phasePair& pair
)
:
    wallLubricationModel(dict, pair),
    Cw1_("Cw1", dimless, dict),
    Cw2_("Cw2", dimless, dict),
    height_lim_("height_lim", dimless, dict),
    height_dir_("height_dir", dimless, dict)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::wallLubricationModels::Antal_limited::~Antal_limited()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volVectorField> Foam::wallLubricationModels::Antal_limited::Fi() const
{
    volVectorField Ur(pair_.Ur());

    const volVectorField& n(nWall());

    Foam::tmp<Foam::volVectorField> Fi = zeroGradWalls
    (
        max
        (
            dimensionedScalar(dimless/dimLength, 0),
            Cw1_/pair_.dispersed().d() + Cw2_/yWall()
        )
       *pair_.continuous().rho()
       *magSqr(Ur - (Ur & n)*n)
       *n
    );
    volVectorField centres = Fi.ref().mesh().C();
    int dir = int (height_dir_.value());
    forAll(centres, cellI) {
        if (centres[cellI][dir]>height_lim_.value()) {
            Fi.ref()[cellI][0] = 0.0;
            Fi.ref()[cellI][1] = 0.0;
            Fi.ref()[cellI][2] = 0.0;
        }
    }

    return Fi;
}


// ************************************************************************* //
