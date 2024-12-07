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

Class
    Foam::dragModels::Grace_limited

Description
    Ishii and Zuber (1979) drag model for dense dispersed bubbly flows.

    Reference:
    \verbatim
        "Drag Coefficient and relative velocity in bubbly, droplet and
         particulate flows",
        Ishii, M., Zuber, N.,
        AIChE Journal 5, Vol. 25, 1979, pp. 843-855.
    \endverbatim

SourceFiles
    Grace_limited.C
    Added by M Rahimi and H Sitaraman, NREL

\*---------------------------------------------------------------------------*/

#ifndef Grace_limited_H
#define Grace_limited_H

#include "dragModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class phasePair;

namespace dragModels
{

/*---------------------------------------------------------------------------*\
                           Class Grace_limited Declaration
\*---------------------------------------------------------------------------*/

class Grace_limited
:
    public dragModel
{

        //- Residual Reynolds Number
        const dimensionedScalar residualRe_;
        //- Where to zero out
        const dimensionedScalar height_lim_;
        const dimensionedScalar height_dir_;
 
public:

    //- Runtime type information
    TypeName("Grace_limited");


    // Constructors

        //- Construct from a dictionary and a phase pair
        Grace_limited
        (
            const dictionary& dict,
            const phasePair& pair,
            const bool registerObject
        );


    //- Destructor
    virtual ~Grace_limited();


    // Member Functions

        //- Drag coefficient
        virtual tmp<volScalarField> CdRe() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace dragModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
