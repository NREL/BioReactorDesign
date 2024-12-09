/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2014-2015 OpenFOAM Foundation
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

#include "Grace_limited.H"
#include "phasePair.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace dragModels
{
    defineTypeNameAndDebug(Grace_limited, 0);
    addToRunTimeSelectionTable(dragModel, Grace_limited, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::dragModels::Grace_limited::Grace_limited
(
    const dictionary& dict,
    const phasePair& pair,
    const bool registerObject
)
:
    dragModel(dict, pair, registerObject),
    residualRe_("residualRe", dimless, dict),
    height_lim_("height_lim", dimless, dict),
    height_dir_("height_dir", dimless, dict)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::dragModels::Grace_limited::~Grace_limited()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::dragModels::Grace_limited::CdRe() const
{
    volScalarField Re(pair_.Re());
    volScalarField Eo(pair_.Eo());
    volScalarField Mo(pair_.Mo());

    volScalarField mud(pair_.dispersed().thermo().mu());
    volScalarField muc(pair_.continuous().thermo().mu());

    volScalarField rhod(pair_.dispersed().rho());
    volScalarField rhoc(pair_.continuous().rho());

    dimensionedScalar mu_ref
    (
        "mu_ref",
        dimensionSet(1,-1,-1,0,0,0,0),
        scalar(0.0009)
    );

    dimensionedScalar g_const
    (
        "g_const",
        dimensionSet(0,1,-2,0,0,0,0),
        scalar(9.81)
    );

    volScalarField H( 4.0/3.0*Eo*pow(Mo,-0.149)*pow((muc/mu_ref),-0.14) );
    H.max(2.0);

    volScalarField J( pos(H-59.3)*(3.42*pow(H,0.441)) + neg(H-59.3)*(0.94*pow(H,0.757)) );

    volScalarField UT( (muc/rhoc/pair_.dispersed().d())* pow(Mo,-0.149)* (J-0.857) );

    volScalarField CdEllipse
    (
        4.0/3.0*mag(g_const)*pair_.dispersed().d()/sqr(UT) * mag(rhoc-rhod)/rhoc
    );
 
    Re.max(residualRe_);
    
    Foam::tmp<Foam::volScalarField> CdRe = pos(Re - 1000)*0.44*Re
      + neg(Re - 1000)*max(24.0*(1.0 + 0.15*pow(Re, 0.687)) , Re*min(CdEllipse,8.0/3.0) );

    volVectorField centres = CdRe.ref().mesh().C();
    int dir = int (height_dir_.value());
    forAll(centres, cellI) {
        if (centres[cellI][dir]>height_lim_.value()) {
            CdRe.ref()[cellI] = 0.0;
        }
    }
    return CdRe;

}


// ************************************************************************* //
