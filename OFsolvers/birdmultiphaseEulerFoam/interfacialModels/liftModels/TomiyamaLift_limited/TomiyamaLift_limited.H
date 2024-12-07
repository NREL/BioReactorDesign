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

Class
    Foam::liftModels::TomiyamaLift_limited

Description
    Lift model of Tomiyama et al.

    Reference:
    \verbatim
        Tomiyama, A., Tamai, H., Zun, I., & Hosokawa, S. (2002).
        Transverse migration of single bubbles in simple shear flows.
        Chemical Engineering Science, 57(11), 1849-1858.
    \endverbatim

    The coefficient for pow3(EoH) proposed by Tomiyama (2002) has been modified
    to make the model continuous at EoH = 10.7 while maintaining the
    lift coefficient proposed by Tomiyama (2002) when EoH >= 10.7.

SourceFiles
    TomiyamaLift_limited.C

\*---------------------------------------------------------------------------*/

#ifndef TomiyamaLift_limited_H
#define TomiyamaLift_limited_H

#include "liftModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class phasePair;

namespace liftModels
{

/*---------------------------------------------------------------------------*\
                        Class TomiyamaLift_limited Declaration
\*---------------------------------------------------------------------------*/

class TomiyamaLift_limited
:
    public liftModel
{
    //- Where to zero out
    const dimensionedScalar height_lim_;
    const dimensionedScalar height_dir_;

public:

    //- Runtime type information
    TypeName("Tomiyama_limited");


    // Constructors

        //- Construct from a dictionary and a phase pair
        TomiyamaLift_limited
        (
            const dictionary& dict,
            const phasePair& pair
        );


    //- Destructor
    virtual ~TomiyamaLift_limited();


    // Member Functions

        //- Lift coefficient
        virtual tmp<volScalarField> Cl() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace liftModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
