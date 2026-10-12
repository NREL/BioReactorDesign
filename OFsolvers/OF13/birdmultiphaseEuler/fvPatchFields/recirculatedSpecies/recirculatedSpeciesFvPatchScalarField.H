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

Class
    Foam::recirculatedSpeciesFvPatchScalarField

Description
    This boundary condition provides a recirculatedSpecies condition,
    calculated as:

        \f[
            Y = Y_{0}*(1-r) *Y_{r}
        \f]
    and
        \f[
            Y_{r} = sum(\alphaRhoPhi_{pr} Y_{pr}) / sum(\alphaRhoPhi_{pr})
        \f]

    where
    \vartable
        r      | recirculation factor [-]
        Y_{0}  | base patch value [-]
        Y_{r}  | recirculating patch mean [-]
        alphaRhoPhi_{pr}| mass flow rate at recirculating patch [Kg/s]
        Y_{pr}| mass fraction at recirculating patch [Kg/Kg]
    \endtable

Usage
    \table
        Property   | Description                  | Req'd? | Default
        Y0         | base value                   | yes    |
        r          | recirculation factor         | yes    |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type       recirculatedSpecies;
        Y0         1; //Initial stream only has this species
        r          table (
                             (0 0)
                             (1 1)
                          ); // Switch to full recirculation after one second
        value      uniform 1; // optional initial value
    }
    \endverbatim

SourceFiles
    recirculatedSpeciesFvPatchScalarField.C

\*---------------------------------------------------------------------------*/

#ifndef recirculatedSpeciesFvPatchScalarField_H
#define recirculatedSpeciesFvPatchScalarField_H

#include "fixedValueFvPatchFields.H"
#include "Function1.H"
#include "phaseModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                          Class recirculatedSpeciesFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class recirculatedSpeciesFvPatchScalarField
:
    public fixedValueFvPatchScalarField
{
    // Private Data

    
        //- Base value
        autoPtr<Function1<scalar>> Y0_;

        //- Recirculation factor
        autoPtr<Function1<scalar>> r_;

        const phaseModel& phase_;

    // Private Member Functions

        //- Return current time
        scalar t() const;


public:

    //- Runtime type information
    TypeName("recirculatedSpecies");


    // Constructors

        //- Construct from patch, internal field and dictionary
        recirculatedSpeciesFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given fixedValueTypeFvPatchField
        //  onto a new patch
        recirculatedSpeciesFvPatchScalarField
        (
            const recirculatedSpeciesFvPatchScalarField&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fieldMapper&
        );

        //- Disallow copy without setting internal field reference
        recirculatedSpeciesFvPatchScalarField
        (
            const recirculatedSpeciesFvPatchScalarField&
        ) = delete;

        //- Copy constructor setting internal field reference
        recirculatedSpeciesFvPatchScalarField
        (
            const recirculatedSpeciesFvPatchScalarField&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchScalarField> clone
        (
            const DimensionedField<scalar, volMesh>& iF
        ) const
        {
            return tmp<fvPatchScalarField>
            (
                new recirculatedSpeciesFvPatchScalarField
                (
                    *this,
                    iF
                )
            );
        }


    // Member Functions

        // Mapping functions

            //- Map the given fvPatchField onto this fvPatchField
            virtual void map(const fvPatchScalarField&, const fieldMapper&);

            //- Reset the fvPatchField to the given fvPatchField
            //  Used for mesh to mesh mapping
            virtual void reset(const fvPatchScalarField&);


        // Evaluation functions

            //- Update the coefficients associated with the patch field
            virtual void updateCoeffs();


        //- Write
        virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
