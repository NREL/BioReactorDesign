/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2015-2023 OpenFOAM Foundation
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
    Foam::BioReactingPhaseModel

Description
    Class which represents phases with volumetric enzymatic reactions. Returns the
    reaction rate and heat using the Michaelis-Menten reaction rate for enzymatic reactions.

    Reference:
    \verbatim
        Michaelis, L., & Menten, M. L. (1913).
        Die Kinetik der InwertinWirkung.
        Biochem., (49), 333-369.
    \endverbatim

    Compared to the reactionRate implementation in OpenFOAM, this version allows
    easier model customization.

Author
    Federico Municchi

SourceFiles
    BioReactingPhaseModel.C

\*---------------------------------------------------------------------------*/

#ifndef BioReactingPhaseModel_H
#define BioReactingPhaseModel_H

#include "phaseModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


/*---------------------------------------------------------------------------*\
                     Class BioReactingPhaseModel Declaration
\*---------------------------------------------------------------------------*/

template<class BasePhaseModel>
class BioReactingPhaseModel
:
    public BasePhaseModel
{
protected:

    // Protected data

        dictionary  dict_;

        // species
        wordList species_; 

        // Maximum uptake rate
        List<dimensionedScalar> maxUptakeRate_;

        // Half saturation constant
        List<dimensionedScalar> K_;

        // Uptake rates Sp
        PtrList<volScalarField> URSp_;

        // Dry-cell Weight
        //volScalarField DCW_;

public:

    // Constructors

        BioReactingPhaseModel
        (
            const phaseSystem& fluid,
            const word& phaseName,
            const bool referencePhase,
            const label index
        );


    //- Destructor
    virtual ~BioReactingPhaseModel();


    // Member Functions

        //- Correct the reaction rates
        virtual void correctReactions();

        //- Return the consumption rate for a specie
        virtual tmp<volScalarField::Internal> R(const label speciei) const;

        //- Return the fuel consumption rate matrix
        virtual tmp<fvScalarMatrix> R(volScalarField& Yi) const;

        //- Return heat release rate (dummy)
        virtual tmp<volScalarField> Qdot() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "BioReactingPhaseModel.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
