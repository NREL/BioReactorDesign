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
    upon reaching a steady value for the phase center of mass.

    Steady state is computed by comparing the average center of mass from
    the previous "2*nsamples" samples against the previous "samples" samples
    (see input below). If the relative difference falls below a user-defined 
    tolerance, steady stated is decleared. 

    Example of function object specification:
    \verbatim
    disengagement
    {
        type            disengagement;
        libs            ("libbirdmultiphaseEulerFoamFunctionObjects.so");

        //- Properties common to all functionObjects
        executeControl  runTime;
        executeInterval 0.01;
        writeControl    runTime;
        writeInterval   0.1; 

        //- Properties specific of this functionobject
        phase           liquid;
        inletPhase      gas;
        inlet           inlet;
        tolerance       1e-2;
        direction       (0 1 0);
        nsamples        50;
        disengage       true;   
    }
    \endverbatim

Usage
    \table
        Property     | Description             | Required    | Default value
        type         | type name: disengagement     | yes         |
        phase        | name of the phase for computing the center of mass     | yes         |
        inletPhase   | name of the phase you want to disengage | yes |
        inlet        | name of the inlet patch | yes |
        tolerance    | relative difference to declare disengagement | yes |
        direction    | normally the direction of gravity | yes |
        nsamples     | number of samples for averaging | yes |
        disengage    | set false for disabling disengaging | yes |
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

        //- Name of phase
        word inletPhaseName_;

        //- Name of inlet patch
        word inletPatch_;

        //- Tolerance for disengaging
        scalar tolerance_;

        //- Number of samples for averaging
        label nsamples_;

        //- Direction for computing the holdup
        vector direction_;

        //- Optional for not disengaging
        bool disengage_;

        //- Vector with phase ceneter of mass and times
        List<Pair<scalar>> phase_com_;

        //- Check if disengaged
        bool disengaged_;

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

        virtual wordList fields() const { return wordList(); };
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //