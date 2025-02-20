/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2018-2021 OpenFOAM Foundation
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

#include "Laakkonen_limited.H"
#include "addToRunTimeSelectionTable.H"
#include "phaseDynamicMomentumTransportModel.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace diameterModels
{
namespace breakupModels
{
    defineTypeNameAndDebug(Laakkonen_limited, 0);
    addToRunTimeSelectionTable
    (
        breakupModel,
        Laakkonen_limited,
        dictionary
    );
}
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::diameterModels::breakupModels::Laakkonen_limited::
Laakkonen_limited
(
    const populationBalanceModel& popBal,
    const dictionary& dict
)
:
    breakupModel(popBal, dict),
    efficiency_
    (
        dimensionedScalar::lookupOrDefault
        (
            "efficiency",
            dict,
            dimless,
            1.0
        )
    ),
    C1_
    (
        dimensionedScalar::lookupOrDefault
        (
            "C1",
            dict,
            dimensionSet(0, -2.0/3.0, 0, 0, 0),
            2.25
        )
    ),
    C2_(dimensionedScalar::lookupOrDefault("C2", dict, dimless, 0.04)),
    C3_(dimensionedScalar::lookupOrDefault("C3", dict, dimless, 0.01)),
    height_lim_(readScalar(dict.lookup("height_lim"))),
    height_dir_(dict.lookup("height_dir"))
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::diameterModels::breakupModels::Laakkonen_limited::setBreakupRate
(
    volScalarField& breakupRate,
    const label i
)
{
    const phaseModel& continuousPhase = popBal_.continuousPhase();
    const sizeGroup& fi = popBal_.sizeGroups()[i];
    breakupRate =
       efficiency_ * (C1_*cbrt(popBal_.continuousTurbulence().epsilon())
       *erfc
        (
            sqrt
            (
                C2_*popBal_.sigmaWithContinuousPhase(fi.phase())
               /(
                    continuousPhase.rho()*pow(fi.dSph(), 5.0/3.0)
                   *pow(popBal_.continuousTurbulence().epsilon(), 2.0/3.0)
                )
            //   + C3_*continuousPhase.thermo().mu()
            //    /(
            //         sqrt(continuousPhase.rho()*fi.phase().rho())
            //        *cbrt(popBal_.continuousTurbulence().epsilon())
            //        *pow(fi.dSph(), 4.0/3.0)
            //     )
            )
        ));
    
    volVectorField centres = breakupRate.ref().mesh().C();
    forAll(centres, cellI) {
        if ( (centres[cellI] & height_dir_) > height_lim_) {
            breakupRate.ref()[cellI] = 0.0;
        }
    } 
}


// ************************************************************************* //
