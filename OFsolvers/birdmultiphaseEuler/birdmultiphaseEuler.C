/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
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

#include "birdmultiphaseEuler.H"
#include "localEulerDdtScheme.H"
#include "surfaceFields.H"
#include "fvcDiv.H"
#include "fvcSurfaceIntegrate.H"
#include "fvcMeshPhi.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{
    defineTypeNameAndDebug(birdmultiphaseEuler, 0);
    addToRunTimeSelectionTable(solver, birdmultiphaseEuler, fvMesh);
}
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

bool Foam::solvers::birdmultiphaseEuler::read()
{
    fluidSolver::read();

    predictMomentum =
        pimple.dict().lookupOrDefault<bool>("momentumPredictor", false);

    faceMomentum =
        pimple.dict().lookupOrDefault<Switch>("faceMomentum", false);

    dragCorrection =
        pimple.dict().lookupOrDefault<Switch>("dragCorrection", false);

    nEnergyCorrectors =
        pimple.dict().lookupOrDefault<int>("nEnergyCorrectors", 1);

    alphaControls.read(mesh.solution().solverDict("alpha"));

    return true;
}


void Foam::solvers::birdmultiphaseEuler::correctCoNum()
{
    scalarField sumPhi
    (
        fvc::surfaceSum(mag(phi))().primitiveField()
    );

    forAll(movingPhases, movingPhasei)
    {
        sumPhi = max
        (
            sumPhi,
            fvc::surfaceSum(mag(movingPhases[movingPhasei].phi()))()
           .primitiveField()
        );
    }

    CoNum_ = 0.5*gMax(sumPhi/mesh.V().primitiveField())*runTime.deltaTValue();

    const scalar meanCoNum =
        0.5
       *(gSum(sumPhi)/gSum(mesh.V().primitiveField()))
       *runTime.deltaTValue();

    Info<< "Courant Number mean: " << meanCoNum
        << " max: " << CoNum << endl;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::solvers::birdmultiphaseEuler::birdmultiphaseEuler(fvMesh& mesh)
:
    fluidSolver(mesh),

    predictMomentum
    (
        pimple.dict().lookupOrDefault<Switch>("momentumPredictor", false)
    ),

    faceMomentum
    (
        pimple.dict().lookupOrDefault<Switch>("faceMomentum", false)
    ),

    dragCorrection
    (
        pimple.dict().lookupOrDefault<Switch>("dragCorrection", false)
    ),

    useTemperaturePredictor
    (
        pimple.dict().lookupOrDefault<Switch>("temperaturePredictor", true)
    ),

    nEnergyCorrectors
    (
        pimple.dict().lookupOrDefault<int>("nEnergyCorrectors", 1)
    ),

    trDeltaT
    (
        LTS
      ? new volScalarField
        (
            IOobject
            (
                fv::localEulerDdt::rDeltaTName,
                runTime.name(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(dimless/dimTime, 1),
            extrapolatedCalculatedFvPatchScalarField::typeName
        )
      : nullptr
    ),

    trDeltaTf
    (
        LTS && faceMomentum
      ? new surfaceScalarField
        (
            IOobject
            (
                fv::localEulerDdt::rDeltaTfName,
                runTime.name(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(dimless/dimTime, 1)
        )
      : nullptr
    ),

    buoyancy(mesh),

    fluid_(mesh),

    phases_(fluid_.phases()),

    movingPhases_(fluid_.movingPhases()),

    phi_(fluid_.phi()),

    momentumTransferSystem_(fluid_),

    heatTransferSystem_(fluid_),

    populationBalanceSystem_(fluid_),

    p_(movingPhases_[0].fluidThermo().p()),

    p_rgh_(buoyancy.p_rgh),

    pressureReference
    (
        p_,
        p_rgh_,
        pimple.dict(),
        fluid_.incompressible()
    ),

    MRF(fluid_.MRF()),

    fluid(fluid_),
    phases(phases_),
    movingPhases(movingPhases_),
    momentumTransfer(momentumTransferSystem_),
    heatTransfer(heatTransferSystem_),
    p(p_),
    p_rgh(p_rgh_),
    phi(phi_),
    rhoCpvs(fluid.thermalPhases().size())
{
    // Read the controls
    read();

    mesh.schemes().setFluxRequired(p_rgh.name());

    if (transient())
    {
        correctCoNum();
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::solvers::birdmultiphaseEuler::~birdmultiphaseEuler()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::birdmultiphaseEuler::preSolve()
{
    if (transient())
    {
        correctCoNum();
    }
    else if (LTS)
    {
        setRDeltaT();
    }

    // Store divU from the previous mesh so that it can be
    // mapped and used in correctPhi to ensure the corrected phi
    // has the same divergence
    if (correctPhi || mesh.topoChanging())
    {
        // Construct and register divU for mapping
        divU = new volScalarField
        (
            "divU0",
            fvc::div(fvc::absolute(phi, movingPhases[0].U()))
        );
    }

    fvModels().preUpdateMesh();

    // Update the mesh for topology change, mesh to mesh mapping
    mesh_.update();
}


void Foam::solvers::birdmultiphaseEuler::prePredictor()
{
    if (pimple.thermophysics() || pimple.flow())
    {
        alphaControls.correct(CoNum);

        fluid_.solve(alphaControls, rAs, momentumTransferSystem_);
        populationBalanceSystem_.solve();

        fluid_.correct();
        populationBalanceSystem_.correct();

        fluid_.correctContinuityError(populationBalanceSystem_.dmdts());
    }
}


void Foam::solvers::birdmultiphaseEuler::momentumTransportPredictor()
{
    fluid_.predictMomentumTransport();
}


void Foam::solvers::birdmultiphaseEuler::thermophysicalTransportPredictor()
{
    // Moved inside the nEnergyCorrectors loop in thermophysicalPredictor()
    // fluid_.predictThermophysicalTransport();
}


void Foam::solvers::birdmultiphaseEuler::momentumTransportCorrector()
{
    fluid_.correctMomentumTransport();
}


void Foam::solvers::birdmultiphaseEuler::thermophysicalTransportCorrector()
{
    fluid_.correctThermophysicalTransport();
}


void Foam::solvers::birdmultiphaseEuler::postSolve()
{
    divU.clear();
}


// ************************************************************************* //
