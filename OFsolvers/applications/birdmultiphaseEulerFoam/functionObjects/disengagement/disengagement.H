/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2020 OpenFOAM Foundation
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
    Foam::functionObjects::disengagement

Description
    This functionObject disengages (i.e., sets the inlet velocity to zero)
    upon reaching a steady holdup.

Usage
    \table
        Property     | Description             | Required    | Default value
        type         | type name: disengagement     | yes         |
    \endtable

See also
    Foam::functionObjects::fvMeshFunctionObject
    Foam::functionObject

SourceFiles
    disengagement.C

\*---------------------------------------------------------------------------*/

#ifndef functionObjects_disengagement_H
#define functionObjects_disengagement_H

#include "fvMeshFunctionObject.H"
#include "phaseSystem.H"
#include "logFiles.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                          Class disengagement Declaration
\*---------------------------------------------------------------------------*/

class disengagement
:
    public fvMeshFunctionObject,
    public logFiles
{
protected:

    // Protected data

        //- Constant access to phases
        const phaseSystem::phaseModelList& phases_;

        //- Name of phase
        word phaseName_;

        //- Name of inlet patch
        word inletPatch_;

        //- Vector with holdups and times
        List<Pair<scalar>> holdup_;

        //- Tolerance for disengaging
        scalar tolerance_;

        //- Number of samples for averaging
        label nsamples_;

        //- Check if disengaged
        bool disengaged_;

        //- Book-keeping for writing array
        label writtenAt_;

        //- Output file header information
        virtual void writeFileHeader(const label i);


public:

    //- Runtime type information
    TypeName("disengagement");


    // Constructors

        //- Construct from Time and dictionary
        disengagement
        (
            const word& name,
            const Time& runTime,
            const dictionary&
        );

        //- Disallow default bitwise copy construction
        disengagement(const disengagement&) = delete;


    //- Destructor
    virtual ~disengagement();


    // Member Functions

        //- Calculate the force fields
        virtual bool execute();

        //- Write the force fields
        virtual bool write();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const disengagement&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
