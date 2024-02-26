/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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

Typedef
    Foam::scalar

Description
    Single floating point number identical to float or double depending on
    whether WM_SP, WM_DP or WM_LP is defined.

SourceFiles
    scalar.C

\*---------------------------------------------------------------------------*/

#ifndef bdoMathFuncs_H
#define bdoMathFuncs_H

#include "floatScalar.H"
#include "doubleScalar.H"
#include "longDoubleScalar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace Foam
{
    //- Normalized upper incomplete gamma function
    scalar incGammaRatio_Q(const scalar a, const scalar x);

    //- Normalized lower incomplete gamma function
    scalar incGammaRatio_P(const scalar a, const scalar x);

    //- Upper incomplete gamma function
    scalar incGamma_Q(const scalar a, const scalar x);

    //- Lower incomplete gamma function
    scalar incGamma_P(const scalar a, const scalar x);
}

#endif
// ************************************************************************* //
