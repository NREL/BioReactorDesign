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
    Foam::functionObjects::kLA

Description
    Computes and phase averaged local and global kLA, as well as the
    average concentration and saturation concentration.

    Example of function object specification:
    \verbatim
    kLA
    {
        type            kLA;
        libs            ("libbirdmultiphaseEulerFunctionObjects.so");

        //- Properties common to all functionObjects
        executeControl  runTime;
        executeInterval 0.01;
        writeControl    runTime;
        writeInterval   0.1;

        //- Properties specific of this functionobject
        continuumPhase  liquid;
        dispersedPhase   gas;
        species         ("CO2" "H2");
        phaseMin        0.05;
        interfaceMax    0.7;

    }
    \endverbatim

Usage
    \table
        Property     | Description             | Required    | Default value
        type         | type name: kLA     | yes         |
        continuumPhase        | name of the phase where the species are dissolved     | yes         |
        dispersedPhase        | name of the dispersed phase     | yes         |
        species      | name of the species | yes |
        phaseMin     | minimum value of the phase to average | yes |
        interfaceMax | maximum value for the interface field | no | -1
    \endtable

See also
    Foam::functionObjects::fvMeshFunctionObject
    Foam::functionObject

SourceFiles
    kLA.C

\*---------------------------------------------------------------------------*/

#ifndef functionObjects_kLA_H
#define functionObjects_kLA_H

#include "fvMeshFunctionObject.H"
#include "phaseSystem.H"
#include "logFiles.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                          Class kLA Declaration
\*---------------------------------------------------------------------------*/

class kLA
:
    public fvMeshFunctionObject,
    public logFiles
{
protected:

    // Protected data

        //- Constant access to phases
        const phaseSystem::phaseModelList& phases_;

        //- Name of phase
        word continuumPhase_;

        //- Name of other phase
        word dispersedPhase_;

        //- Name of species
        wordList species_;

        //- Phase minimum value
        scalar phaseMin_;

        //- Interface maximum value
        scalar interfaceMax_;

        //- Holds current data
        scalarField data_;

        //- Output file header information
        virtual void writeFileHeader(const label i);


public:

    //- Runtime type information
    TypeName("kLA");


    // Constructors

        //- Construct from Time and dictionary
        kLA
        (
            const word& name,
            const Time& runTime,
            const dictionary&
        );

        //- Disallow default bitwise copy construction
        kLA(const kLA&) = delete;


    //- Destructor
    virtual ~kLA();


    // Member Functions

        //- Calculate the force fields
        virtual bool execute();

        //- Write the force fields
        virtual bool write();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const kLA&) = delete;

        virtual wordList fields() const { return species_; };
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
