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
#include "fvcDdt.H"
#include "fvcDiv.H"
#include "fvcSup.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmSup.H"

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

void Foam::solvers::birdmultiphaseEuler::compositionPredictor()
{
    autoPtr<HashPtrTable<fvScalarMatrix>> popBalSpecieTransferPtr =
        populationBalanceSystem_.specieTransfer();
    HashPtrTable<fvScalarMatrix>& popBalSpecieTransfer =
        popBalSpecieTransferPtr();

    fluid_.correctReactions();

    forAll(fluid.multicomponentPhases(), multicomponentPhasei)
    {
        phaseModel& phase = fluid_.multicomponentPhases()[multicomponentPhasei];

        UPtrList<volScalarField>& Y = phase.YRef();
        const volScalarField& alpha = phase;
        const volScalarField& rho = phase.rho();

        forAll(Y, i)
        {
            if (phase.solveSpecie(i))
            {
                fvScalarMatrix YiEqn
                (
                    phase.YiEqn(Y[i])
                 ==
                    *popBalSpecieTransfer[Y[i].name()]
                  + fvModels().source(alpha, rho, Y[i])
                );

                YiEqn.relax();

                fvConstraints().constrain(YiEqn);

                YiEqn.solve("Yi");

                fvConstraints().constrain(Y[i]);
            }
            else
            {
                Y[i].correctBoundaryConditions();
            }
        }
    }

    fluid_.correctSpecies();
}


void Foam::solvers::birdmultiphaseEuler::energyPredictor()
{
    autoPtr<HashPtrTable<fvScalarMatrix>> heatTransferPtr;
    
    if ( useTemperaturePredictor )
    {
        heatTransferPtr = heatTransferSystem_.heatTransferT();
    }
    else
    {
        heatTransferPtr = heatTransferSystem_.heatTransfer();
    }
    
    HashPtrTable<fvScalarMatrix>& heatTransfer =
        heatTransferPtr();

    autoPtr<HashPtrTable<fvScalarMatrix>> popBalHeatTransferPtr =
        populationBalanceSystem_.heatTransfer();
    HashPtrTable<fvScalarMatrix>& popBalHeatTransfer =
        popBalHeatTransferPtr();

    forAll(fluid.thermalPhases(), thermalPhasei)
    {
        phaseModel& phase = fluid_.thermalPhases()[thermalPhasei];

        const volScalarField& alpha = phase;
        const volScalarField& rho = phase.rho();

        
        if( useTemperaturePredictor )
        {

            volScalarField& he = phase.thermo().he();
            volScalarField& T = phase.thermo().T();
            const volScalarField Told = T;
            
            if ( rhoCpvs[thermalPhasei].empty() )
            {
                rhoCpvs.set
                (
                    thermalPhasei,
                    rho*phase.thermo().Cpv()
                );
            }
            else
            {
                volScalarField& rhoCpv = rhoCpvs[thermalPhasei];
                rhoCpv =  rho*phase.thermo().Cpv();
            }
            
           
            fvScalarMatrix TEqn
            (
                fvm::ddt(rhoCpvs[thermalPhasei], T)
            ==
                *heatTransfer[phase.name()]
            );

            TEqn.relax();
            TEqn.solve();
            fvConstraints().constrain(phase.thermo().T());

            he += phase.thermo().Cpv() * (T - Told);
            he.correctBoundaryConditions();
        }

        fvScalarMatrix EEqn
        (
            phase.heEqn()
         ==
           *popBalHeatTransfer[phase.name()]
          + fvModels().source(alpha, rho, phase.thermo().he())
        );

        EEqn.relax();
        fvConstraints().constrain(EEqn);
   
        if ( useTemperaturePredictor )
        {
            solve( EEqn == fvc::ddt(rho, phase.thermo().he()) );
        }
        else
        {
            solve( EEqn == *heatTransfer[phase.name()] );
        }

        EEqn.solve();
        fvConstraints().constrain(phase.thermo().he());
    }

    fluid_.correctThermo();
    fluid_.correctContinuityError(populationBalanceSystem_.dmdts());
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::birdmultiphaseEuler::thermophysicalPredictor()
{
    for (int Ecorr=0; Ecorr<nEnergyCorrectors; Ecorr++)
    {
        fluid_.predictThermophysicalTransport();
        compositionPredictor();
        energyPredictor();

        forAll(fluid.thermalPhases(), thermalPhasei)
        {
            const phaseModel& phase = fluid.thermalPhases()[thermalPhasei];

            Info<< phase.name() << " min/max T "
                << min(phase.thermo().T()).value()
                << " - "
                << max(phase.thermo().T()).value()
                << endl;
        }
    }
}


// ************************************************************************* //
