double gradfunMix(double V1, double V2){
    return 3.0*V2*V2 + 2.0*V1*V2 - V1*V1;
}
double funMix(double V1, double V2, double P, double rhoL, double A){
    return V2*V2*V2 + V1*V2*V2 - V2*V1*V1 - V1*V1*V1 - 4.0*P/(rhoL * A);
}

double findV2(double P, double rhoL, double A, double V1) {
    int newton_iter = 100;
    double V2 = 2*V1;
    double V2_old;
    double V2_new;
    for (int i=0; i<newton_iter; i++) {
        V2_old = V2;
	//std::cout << "funMix(V1, V2, P, rhoL, A) = " << funMix(V1, V2, P, rhoL, A) << "\n";
	//std::cout << "gradfunMix(V1,V2) = " << gradfunMix(V1,V2) << "\n";
        
        V2 = V2_old - funMix(V1, V2, P, rhoL, A) / gradfunMix(V1,V2);
        V2_new = V2;
        if (std::abs(V2_new - V2_old) < 1e-12) {
            break;
        }
    }
    std::cout << "V2 = " << V2 << "\n";
    std::cout << "V1 = " << V1 << "\n";
    return V2;
}
