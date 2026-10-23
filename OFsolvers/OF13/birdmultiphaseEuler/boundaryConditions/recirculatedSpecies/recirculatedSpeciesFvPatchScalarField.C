/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026 OpenFOAM Foundation
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

#include "recirculatedSpeciesFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "fieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::scalar Foam::recirculatedSpeciesFvPatchScalarField::t() const
{
    return db().time().value();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::recirculatedSpeciesFvPatchScalarField::
recirculatedSpeciesFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchScalarField(p, iF),
    Y0_
    (
        Function1<scalar>::New
        (
            "Y0",
            db().time().userUnits(),
            unitAny,
            dict
        )
    ),
    r_
    (
        Function1<scalar>::New
        (
            "r",
            db().time().userUnits(),
            unitAny,
            dict
        )
    ),
    phaseName_(dict.lookupOrDefault<word>("phase","gas")),
    otherPatchName_(dict.lookup("otherPatch"))
{


//    fixedValueFvPatchScalarField::evaluate();

    
    // Initialise with the value entry if evaluation is not possible
    fvPatchScalarField::operator=
    (
        scalarField("value", iF.dimensions(), dict, p.size())
    );
    
}


Foam::recirculatedSpeciesFvPatchScalarField::
recirculatedSpeciesFvPatchScalarField
(
    const recirculatedSpeciesFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fieldMapper& mapper
)
:
    fixedValueFvPatchScalarField(ptf, p, iF, mapper),
    Y0_(ptf.Y0_, false),
    r_(ptf.r_, false),
    phaseName_(ptf.phaseName_),
    otherPatchName_(ptf.otherPatchName_)
{}


Foam::recirculatedSpeciesFvPatchScalarField::
recirculatedSpeciesFvPatchScalarField
(
    const recirculatedSpeciesFvPatchScalarField& ptf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedValueFvPatchScalarField(ptf, iF),
    Y0_(ptf.Y0_, false),
    r_(ptf.r_, false),
    phaseName_(ptf.phaseName_),
    otherPatchName_(ptf.otherPatchName_)
{}



// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::recirculatedSpeciesFvPatchScalarField::map
(
    const fvPatchScalarField& ptf,
    const fieldMapper& mapper
)
{
    fixedValueFvPatchScalarField::map(ptf, mapper);
}


void Foam::recirculatedSpeciesFvPatchScalarField::reset
(
    const fvPatchScalarField& ptf
)
{
    fixedValueFvPatchScalarField::reset(ptf);
}


void Foam::recirculatedSpeciesFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const phaseModel& phase
    (
        db().lookupObject<phaseModel>
        (
            IOobject::groupName("alpha", phaseName_)
        )
    );


    if ( !db().foundObject<surfaceScalarField>("alphaRhoPhi." + phase.name()) )
    {
        return;
    }

    // Since we're inside initEvaluate/evaluate there might be processor
    // comms underway. Change the tag we use.
    int oldTag = UPstream::msgType();
    UPstream::msgType() = oldTag+1;

    const fvMesh& mesh = this->internalField().mesh();

    label patchID = mesh.boundaryMesh().findIndex(otherPatchName_);

    scalar alphaRhoPhiYr = 0.;
    scalar alphaRhoPhiTot = 1e-16;
    
    if ( patchID != 1 )
    {
        const fvPatch& nbrPatch = mesh.boundary()[patchID];
        const scalarField& alphaRhoPhip = nbrPatch.lookupPatchField<surfaceScalarField, scalar>("alphaRhoPhi." + phase.name());
        const scalarField& Yp = nbrPatch.lookupPatchField<volScalarField, scalar>(this->internalField().name());

        alphaRhoPhiTot = max( mag( sum(alphaRhoPhip)),1e-16 );
        alphaRhoPhiYr = mag( sum(alphaRhoPhip*Yp) );
    }

    reduce(alphaRhoPhiTot, sumOp<scalar>());
    reduce(alphaRhoPhiYr, sumOp<scalar>());
    
    const scalar Yr = min(max( alphaRhoPhiYr/alphaRhoPhiTot, 0.), 1.);

    scalar rf = min(max(r_->value(t()),0.),1.);

    // Restore tag
    UPstream::msgType() = oldTag;

    fixedValueFvPatchScalarField::operator==( (1. - rf)*Y0_->value(t()) + rf*Yr );

    fixedValueFvPatchScalarField::updateCoeffs();
}


void Foam::recirculatedSpeciesFvPatchScalarField::write
(
    Ostream& os
) const
{
    fvPatchScalarField::write(os);
    writeEntry(os, db().time().userUnits(), unitAny, Y0_());
    writeEntry(os, db().time().userUnits(), unitAny, r_());
    writeEntry(os, "phase", phaseName_);
    writeEntry(os, "otherPatch", otherPatchName_);
    writeEntry(os, "value", *this);
}


// * * * * * * * * * * * * * * Build Macro Function  * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchScalarField,
        recirculatedSpeciesFvPatchScalarField
    );
}

// ************************************************************************* //
