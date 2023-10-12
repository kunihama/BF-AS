#include<oxstd.h>
#include<oxprob.h>
#include<oxfloat.h>
#include<oxdraw.h>

main()
{											
	decl time = timer();
	decl T, K, H, P, Q, L, i, k, p, q, j, l, h, r, R, B, b, d, m, s, t;
	decl nsim, nburn, msamp, cn, da, vs, mmu, th, ms, vmu, vsig2, mh; 
	decl mL, vh0, vf, mb, meta, mG, mSig, mM, mP, ct, vy0, vi, vyt, mst, mit;
    decl kk, vpi, amLam, vpi0, vphi0, vphi1, vh, dk, vy, vsig, mz, mD, mys;
	decl md0, md1, vpi1, vh1, mh_s, mh0, mh1;
	decl K_s, mstore, mtest, mtraining;
	
	/*--- Seed ---*/											

	ranseed(11); 	   // set a positive integer	

	/*--- set sampling option ---*/

	L = 5;					// # of causes in the data	
	K_s = 10; 			    // # of factors
	nsim = 500;			    // # of collected MCMC iteration
	th = 20;				// # of skimming, nsim x th = total # of MCMC iteration
	nburn = 1000;			// burn-in period	
	R = 200; 				// # of Monte Carlo simulation for evaluation of P(x|y)
	T = 5;					// # of folds in cross validation

	/*--- load data ---*/

	mtest = loadmat("test.csv"); // test data
	mtraining = loadmat("training.csv"); // training data

	mstore = zeros(T, K_s);
	vi = sortcindex( ranu(rows(mtraining), 1) );
	mtraining = mtraining[vi][];	
	d = floor(rows(mtraining) / T);
	
	for(t = 1; t <= T+1; t++){

	  /*--- test & traning data for cross validation ---*/

	  if(t == T+1){ mD = mtest; mb = mtraining; }
	  else{		
		if(t < T){ vi = range( d*(t-1), (d*t-1) ); }
		else{ vi = range( d*(t-1), (rows(mtraining)-1) ); }

		vs = dropc(range(0, rows(mtraining)-1), vi);
		mD = mtraining[vi][];
		mb = mtraining[vs][];
	   }// else
	
	vy0 = vy = mD[][0];
	ms = mD[][1:];
	mM = (ms .== 999);				// 999: missing data
	cn = rows(vy);
	P = columns(ms);

    mD = mb;	
	vyt = mD[][0];
	mst = mD[][1:];
	mit = (mst .== 999); 
	ct  = rows(vyt);
	
	/*--- start loop for factors ---*/

	for(K = 1; K <= K_s; K++){	
	
	if(t == T+1){vh = meanc(mstore); K = sumr( ( vh .== min(vh) ) .* range(1, K_s) ); }

	/*--- Initial values ---*/
	
	amLam = new array[L];	 
	for(l=0 ; l<L ; l++)
	  amLam[l] = zeros(K, P); 

	vphi0 = vphi1 = ones(1, P); 
	mz = ( mst .> 0 ) - ( mst .<= 0 );
	mmu = zeros(L, P);
	meta = rann(ct, K);	
	vpi0 = meanc(vy0 .== range(0, L-1));
	vpi  = meanc(vyt .== range(0, L-1));

    md0 = !mM  .* !ms;  md1 = !mM  .* ms;	
	msamp = <>;		

	/*--- MCMC sampling ---*/
	
	println("\n\nIteration:");

	/*----------- S A M P L I N G   S T A R T -----------*/
		
	for(kk=-nburn ; kk<th*nsim ; kk++){  // MCMC iteration

    /*--- sampling mu ---*/
	
	mL = zeros(ct, P);
	for(l=0 ; l<L ; l++){
      vi = vecindex(vyt .== l);
	  if(rows(vi) > 0)
		mL[vi][] = mz[vi][] - meta[vi][] * amLam[l];
	}// l

	for(j=0 ; j<P ; j++){
	  mh = (vyt .== range(0, L-1)) .* !mit[][j];
	  vsig2 = 1 ./ (sumc(mh)' + vphi0[j]);
	  vmu = vsig2 .* sumc(mL[][j] .* mh)';
	  mmu[][j] = vmu + sqrt(vsig2) .* rann(L, 1);
	}// j

    /*--- sampling lambda ---*/

	for(l=0 ; l<L ; l++){
	  for(j=0 ; j<P ; j++){
		vi = vecindex( (vyt .== l) .* !mit[][j]);
		if(rows(vi) > 0){
		  mSig = invert(vphi1[j]*unit(K) + meta[vi][]' meta[vi][]);
		  vmu = mSig * meta[vi][]' * (mz[vi][j] - mmu[l][j]);
		  amLam[l][][j] = vmu + choleski(mSig) * rann(K, 1);
		}// if
		else
		  amLam[l][][j] = sqrt(1 ./ vphi1[j]) .* rann(K, 1);
	  }// j
	}// l

    /*--- sampling eta ---*/

	for(i=0 ; i<ct ; i++){
	  vi = vecindex(!mit[i][]);	l = vyt[i];
	  mSig = invert( unit(K) + amLam[l][][vi] * amLam[l][][vi]' );
	  vmu = mSig * amLam[l][][vi] * (mz[i][vi] - mmu[l][vi])';
	  meta[i][] = ( vmu + choleski(mSig) * rann(K, 1) )';
	}// i  

	/*--- sampling tau ---*/
	
	vh = sumc(mmu.^2);
	vphi0 = rangamma(1, P, (1 + L)/2, (1 + vh)/2 );			 	

	/*--- sampling phi ---*/

	vh = zeros(1, P);
	for(l=0 ; l<L ; l++)
	  vh += sumc(amLam[l].^2);
	
	vphi1 = rangamma(1, P, (1 + K*L)/2, (1 + vh)/2 );			 	

		/*--- sampling z ---*/	
	
		mh_s = zeros(ct, P);
		for(l = 0 ; l < L ; l++){
		  vi = vecindex( vyt .== l );
		  mh_s[vi][] = mmu[l][] + meta[vi][] * amLam[l];
	    }// l
		  
		mh = probn( 0 - mh_s );
		mh0 = ( mst .== 0 ) .* mh + ( mst .== 1 ) .* ( 1 - mh ) + ( mst .== 999 );
		mh1 = ( mst .== 1 ) .* mh;
		mz = mh_s + quann( mh0 .* ranu(ct, P) + mh1 );
        if( any( mz .== +.Inf ) + any( mz .== -.Inf ) ){
		  for(j = 0 ; j < P ; j++){
		    vi = vecindex( (mz[][j] .== +.Inf) + (mz[][j] .== -.Inf) );
			if( rows( vi ) > 0 ){ mz[vi][j] = 0.5 * ( (mst[vi][j] .== 1) - (mst[vi][j] .== 0) ); }
		  }// j
		}// if

	/*--- storing sample ---*/

	if(kk >= 0 && !imod(kk, th)){										 

	/*--- estimate distribution in a target site ---*/

	  mh = zeros(cn, L);
	  for(l=0; l<L; l++){
		mL = mmu[l][] + rann(R, K) * amLam[l];
		mP = probn( -mL ); 
		mP += ( (log(mP) .== -.Inf) - (log(1-mP) .== -.Inf) ) .* 10^(-10);
	    mG = md0 * log(mP)' + md1 * log(1 - mP)';	  
		mh[][l] = meanr( exp(mG) );
	  }// l
	  vy = sumr( cumulate( ( mh ./ sumr(mh) )' )' .< ranu(cn, 1) );	
	  vpi = meanc(vy .== range(0, L-1));
	  msamp |= sumr( fabs(vpi0 - vpi) ) ~ vpi; // L1	  

	}// saving samples
	
	/*--- print counter ---*/

	  if(!imod(kk, 1000)){
	    if(t <= T){ println(sprint("cross validation:",t,", # of factor:",K,", MCMC iteration:",kk)); }
	    else{ println(sprint("selected # of factor:",int(K),", MCMC iteration:",kk)); }
	  }	  
										  
	}//kk [MCMC]		 	

	  if(t <= T){ mstore[t-1][K-1] = meanc(msamp[][0]); }
	  else{ K = K_s; }

	}//K [factor]		 	

	}//t [cross-validation]		 	

	/*------------ S A M P L I N G   E N D --------------*/

	/*--- output ---*/

	savemat("BF-result.csv", msamp);

	println("\n\nTime: ", timespan(time));
}
