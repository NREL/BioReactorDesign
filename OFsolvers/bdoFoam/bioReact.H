{
    const vectorField& C = mesh.C();
    std::vector<double> solnvec(microbemodel::nvars);
    int O2id=0;
    double liqvol;
    double dt_react=0.01; //hours (need to have it as an input)
    double mwt_O2=32.0;
    double our_avg;

    
    //in liquid phase
    UPtrList<volScalarField>& Y2 = phase2.YActiveRef();

    forAll(Y2,i)
    {
        Info<<"species names:\t"<<Y2[i].name()<<"\n";
        if(Y2[i].name()=="O2")
        {
            O2id=i;
            break;
        }
    }

    Info << "Starting reaction update..\n";
    Info << "O2 id is = "<<O2id<<"\tspecname:"<<Y2[O2id].name()<<"\n";


    //get averaged values as mixing time scales are small compared to 
    //reactions
    liqvol = fvc::domainIntegrate(alpha2).value();
    microbemodel::X_avg  = fvc::domainIntegrate(alpha2*microbe_conc).value()/liqvol;
    microbemodel::G_avg  = fvc::domainIntegrate(alpha2*glucose).value()/liqvol;
    microbemodel::Xy_avg = fvc::domainIntegrate(alpha2*xylose).value()/liqvol;
    microbemodel::A_avg  = fvc::domainIntegrate(alpha2*acetoin).value()/liqvol;
    microbemodel::B_avg  = fvc::domainIntegrate(alpha2*bdiol).value()/liqvol;
    
    
    //find initial OUR
    forAll(C,i)
    {
        solnvec[microbemodel::X]  = microbemodel::X_avg;
        solnvec[microbemodel::G]  = microbemodel::G_avg;
        solnvec[microbemodel::Xy] = microbemodel::Xy_avg;
        solnvec[microbemodel::A]  = microbemodel::A_avg;
        solnvec[microbemodel::B]  = microbemodel::B_avg;
        solnvec[microbemodel::O2] = rho2[i]*Y2[O2id][i]*1000.0/mwt_O2; //mol/m3
        our[i]=0.5*microbemodel::get_our(solnvec,microbemodel::nvars);
    }
   
    //average OUR
    our_avg=fvc::domainIntegrate(alpha2*our).value()/liqvol;

    if(Pstream::master())
    {
        //this our is the old our that is printed
        os_timehist << reaction_time <<"\t" << microbemodel::X_avg <<
        "\t" << microbemodel::G_avg << "\t" << microbemodel::Xy_avg << "\t" << microbemodel::A_avg <<
        "\t" << microbemodel::B_avg << "\t" << our_avg << "\n";   
        os_timehist.flush(); 
    }

    Info<<"Avg quantities before reaction:"<<microbemodel::X_avg<<"\t"<<microbemodel::G_avg<<"\t"<<
            microbemodel::Xy_avg<<"\t"<<microbemodel::A_avg<<"\t"<<microbemodel::B_avg<<"\t"<<our_avg<<"\n";

    //loop over cells, populate solution vector with average quantities and local O2
    //integrate reactions at each cell to prescribed time
    forAll(C,i)
    {
        solnvec[microbemodel::X]  = microbemodel::X_avg;
        solnvec[microbemodel::G]  = microbemodel::G_avg;
        solnvec[microbemodel::Xy] = microbemodel::Xy_avg;
        solnvec[microbemodel::A]  = microbemodel::A_avg;
        solnvec[microbemodel::B]  = microbemodel::B_avg;
        solnvec[microbemodel::O2] = rho2[i]*Y2[O2id][i]*1000.0/mwt_O2; //mol/m3

        /*Info<<"for cell "<<i<<"++++++++++++++++++++++++++++\n";
        Info<<"solnvec before:"<<solnvec[microbemodel::X]<<"\t"<<solnvec[microbemodel::G]<<"\t"<<
            solnvec[microbemodel::Xy]<<"\t"<<solnvec[microbemodel::A]<<"\t"<<solnvec[microbemodel::B]<<"\t"
            <<solnvec[microbemodel::O2]<<"\n";*/

        microbemodel::advance(solnvec,microbemodel::nvars,0.0,reaction_update_time.value(),dt_react);
        
        /*Info<<"solnvec after:"<<solnvec[microbemodel::X]<<"\t"<<solnvec[microbemodel::G]<<"\t"<<
            solnvec[microbemodel::Xy]<<"\t"<<solnvec[microbemodel::A]<<"\t"<<solnvec[microbemodel::B]<<"\t"
            <<solnvec[microbemodel::O2]<<"\n";
        Info<<"+++++++++++++++++++++++++++++++++++++++\n";*/

        microbe_conc[i] = solnvec[microbemodel::X];
        glucose[i]      = solnvec[microbemodel::G];
        xylose[i]       = solnvec[microbemodel::Xy];
        acetoin[i]      = solnvec[microbemodel::A];
        bdiol[i]        = solnvec[microbemodel::B];
    }
    
    Info << "finished reaction update..\n";

    liqvol = fvc::domainIntegrate(alpha2).value();
    microbemodel::X_avg  = fvc::domainIntegrate(alpha2*microbe_conc).value()/liqvol;
    microbemodel::G_avg  = fvc::domainIntegrate(alpha2*glucose).value()/liqvol;
    microbemodel::Xy_avg = fvc::domainIntegrate(alpha2*xylose).value()/liqvol;
    microbemodel::A_avg  = fvc::domainIntegrate(alpha2*acetoin).value()/liqvol;
    microbemodel::B_avg  = fvc::domainIntegrate(alpha2*bdiol).value()/liqvol;

    //find final OUR
    forAll(C,i)
    {
        solnvec[microbemodel::X]  = microbemodel::X_avg;
        solnvec[microbemodel::G]  = microbemodel::G_avg;
        solnvec[microbemodel::Xy] = microbemodel::Xy_avg;
        solnvec[microbemodel::A]  = microbemodel::A_avg;
        solnvec[microbemodel::B]  = microbemodel::B_avg;
        solnvec[microbemodel::O2] = rho2[i]*Y2[O2id][i]*1000.0/mwt_O2; //mol/m3
        our[i] += 0.5*microbemodel::get_our(solnvec,microbemodel::nvars); 
    }
    our_avg=fvc::domainIntegrate(alpha2*our).value()/liqvol; 
    Info<<"Avg quantities after reaction:"<<microbemodel::X_avg<<"\t"<<microbemodel::G_avg<<"\t"<<
        microbemodel::Xy_avg<<"\t"<<microbemodel::A_avg<<"\t"<<microbemodel::B_avg<<"\t"<<our_avg<<"\n";

    //exit(0);
}
