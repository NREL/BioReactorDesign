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

\*---------------------------------------------------------------------------*/

#include "Henry_limited.H"
#include "phasePair.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace interfaceCompositionModels
{
    defineTypeNameAndDebug(Henry_limited, 0);
    addToRunTimeSelectionTable(interfaceCompositionModel, Henry_limited, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::interfaceCompositionModels::Henry_limited::Henry_limited
(
    const dictionary& dict,
    const phasePair& pair
)
:
    interfaceCompositionModel(dict, pair),
    k_(dict.lookup("k")),
    height_lim_("height_lim", dimless, dict),
    height_dir_("height_dir", dimless, dict),
    YSolvent_
    (
        IOobject
        (
            IOobject::groupName("YSolvent", pair.name()),
            pair.phase1().mesh().time().timeName(),
            pair.phase1().mesh()
        ),
        pair.phase1().mesh(),
        dimensionedScalar(dimless, 1)
    )
{
    if (k_.size() != species().size())
    {
        FatalErrorInFunction
            << "Differing number of species and solubilities"
            << exit(FatalError);
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::interfaceCompositionModels::Henry_limited::~Henry_limited()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::interfaceCompositionModels::Henry_limited::update(const volScalarField& Tf)
{
    YSolvent_ = scalar(1);

    forAllConstIter(hashedWordList, species(), iter)
    {
        YSolvent_ -= Yf(*iter, Tf);
    }
}


Foam::tmp<Foam::volScalarField> Foam::interfaceCompositionModels::Henry_limited::Yf
(
    const word& speciesName,
    const volScalarField& Tf
) const
{
    if (species().found(speciesName))
    {
        const label index = species()[speciesName];

        Foam::tmp<Foam::volScalarField> Yf_limited = k_[index]
           *otherComposition().Y(speciesName)
           *otherThermo().rho()
           /thermo().rho();
  
        volVectorField centres = Yf_limited.ref().mesh().C();
        int dir = int (height_dir_.value());
        forAll(centres, cellI) {
              if (centres[cellI][dir]>height_lim_.value()) {
                  Yf_limited.ref()[cellI] = 0.0;
              }
        } 

        return Yf_limited;
    }
    else
    {
        return YSolvent_*composition().Y(speciesName);
    }
}


Foam::tmp<Foam::volScalarField> Foam::interfaceCompositionModels::Henry_limited::YfPrime
(
    const word& speciesName,
    const volScalarField& Tf
) const
{
    return volScalarField::New
    (
        IOobject::groupName("YfPrime", pair().name()),
        pair().phase1().mesh(),
        dimensionedScalar(dimless/dimTemperature, 0)
    );
}


// ************************************************************************* //
