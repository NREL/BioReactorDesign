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

#include "Grace.H"
#include "addToRunTimeSelectionTable.H"
#include "uniformDimensionedFields.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace dragModels
{
    defineTypeNameAndDebug(Grace, 0);
    addToRunTimeSelectionTable(dragModel, Grace, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::dragModels::Grace::Grace
(
    const dictionary& dict,
    const phaseInterface& interface,
    const bool registerObject
)
:
    dispersedDragModel(dict, interface, registerObject),
    residualRe_("residualRe", dimless, dict),
    muRef_("muRef", dimensionSet(1,-1,-1,0,0,0,0), dict)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::dragModels::Grace::~Grace()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::dragModels::Grace::CdRe() const
{
    volScalarField Re(interface_.Re());
    volScalarField Eo(interface_.Eo());
    volScalarField Mo(interface_.Mo());

    const volScalarField& rhod = interface_.dispersed().rho();
    const volScalarField& rhoc = interface_.continuous().rho();

    const volScalarField nuc = interface_.continuous().fluidThermo().nu();

    const uniformDimensionedVectorField& g =
        interface_.continuous().mesh().lookupObject<uniformDimensionedVectorField>("g");

    volScalarField H( 4.0/3.0*Eo*pow(Mo,-0.149)*pow((nuc*rhoc/muRef_),-0.14) );
    H.max(2.0);

    volScalarField J( pos(H-59.3)*(3.42*pow(H,0.441)) + neg(H-59.3)*(0.94*pow(H,0.757)) );

    volScalarField UT( (nuc/interface_.dispersed().d())* pow(Mo,-0.149)* (J-0.857) );

    volScalarField CdEllipse
    (
        4.0/3.0*mag(g)*interface_.dispersed().d()/sqr(UT) * mag(rhoc-rhod)/rhoc
    );

    Re.max(residualRe_);
    return
        pos(Re - 1000)*0.44*Re
      + neg(Re - 1000)*max(24.0*(1.0 + 0.15*pow(Re, 0.687)) , Re*min(CdEllipse,8.0/3.0) );

}


// ************************************************************************* //
