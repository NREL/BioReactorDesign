/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2015-2020 OpenFOAM Foundation
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
    Foam::interfaceCompositionModels::Henry_limited

Description
    Henry_limited's law for gas solubility in liquid. The concentration of a dissolved
    species in the liquid is proportional to its partial pressure in the gas. A
    dimensionless solubility, \f$k\f$, is given for each species. This is the
    ratio of the concentration of the species in the liquid to the
    corresponding concentration in the gas; i.e., \f$k =
    c_{i,liq}/c_{i,gas}\f$. Mixing in the gas is assumed to be ideal.

SourceFiles
    Henry_limited.C

\*---------------------------------------------------------------------------*/

#ifndef Henry_limited_H
#define Henry_limited_H

#include "interfaceCompositionModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class phasePair;

namespace interfaceCompositionModels
{

/*---------------------------------------------------------------------------*\
                            Class Henry_limited Declaration
\*---------------------------------------------------------------------------*/

class Henry_limited
:
    public interfaceCompositionModel
{
    // Private Data

        //- Dimensionless solubility coefficients
        const scalarList k_;

        //- The remaining solvent species fraction
        volScalarField YSolvent_;

        //- Where to zero out
        const dimensionedScalar height_lim_;
        const dimensionedScalar height_dir_;

public:

    //- Runtime type information
    TypeName("Henry_limited");

    // Constructors

        //- Construct from components
        Henry_limited
        (
            const dictionary& dict,
            const phasePair& pair
        );


    //- Destructor
    virtual ~Henry_limited();


    // Member Functions

        //- Update the composition
        virtual void update(const volScalarField& Tf);

        //- The interface species fraction
        virtual tmp<volScalarField> Yf
        (
            const word& speciesName,
            const volScalarField& Tf
        ) const;

        //- The interface species fraction derivative w.r.t. temperature
        virtual tmp<volScalarField> YfPrime
        (
            const word& speciesName,
            const volScalarField& Tf
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace interfaceCompositionModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
