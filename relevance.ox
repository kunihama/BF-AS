#include<oxstd.h>
#include<oxprob.h>
#include<oxfloat.h>
#import <maximize>

	decl T, K, P, Q, L, i, k, p, q, j, l, r, R, B, b, d, m, s, G, g, t, R2, a;
	decl nsim, nburn, msamp, cn, da, vs, mmu, th, ms, vmu, vsig2, mh, mh0; 
	decl mL, vh0, mb, meta, mG, mSig, mM, mP, ct, vy0, vi, vyt, mst;
    decl kk, vpi, amLam, vpi0, vh, mz, mD, amC, vsex, vage, mx, amB, mh_s;
	decl md0, md1, vh1, vh2, K_s, mstore, mtest, mtraining, vi0, vd, vz, vi1;
	decl vphi_M, vphi_L, mphi_B, mphi_C, mgam, vmis_s, mp_sa, vx, vp, amP, amG;
	decl vsex0, vage0, mx0, mh1, vy, dm_s, dsd_s, dm_a, dsd_a, vmis0_a, vmis_a, vmis0_s;
	decl vpi_y, mdata, amh, dHy, mh2, mh3, mpi0_e, mpi1_e, mpi2_e, mpi3_e, mpi4_e, mP0, mP1, mP2, mP3;
    decl mG0, mG1, mG2, mG3, mpi1_a, mpi2_a, mpi3_a, mpi4_a, vH0, vH1, dH, vy_e, mx_e, vsex_e, vage_e;
	decl vl0, vl1, vl2, vl3, vh3, msamp0, msamp1, vmi, vcmi, dh, ml, mpi0, vl, ml0, ml1, ml2;
	
main()
{											
	decl time = timer();
	
	/*--- Seed ---*/											

	ranseed(11); 	   // set a positive integer	

	/*--- set sampling option ---*/

	L = 5;					// # of causes in the data	
	K_s = 5; 			    // # of factors	in error term
	nsim = 500;			    // # of collected MCMC iteration
	th = 20;				// # of skimming, nsim * th = total # of MCMC iteration
	nburn = 1000;			// burn-in period	
	T = 5;					// # of folds in cross validation
	R = 1000;				// # of Monte Carlo simulation for evaluation of P(x|y,age,sex)
	R2 = 10000;				// # of Monte Carlo simulation for evaluation of CMI
	
	/*--- load data ---*/

	mdata = loadmat("data.csv");
	
	mstore = zeros(T, K_s);
	vi = sortcindex( ranu( rows(mdata), 1 ) );
	mdata = mdata[vi][];	
	d = floor( rows(mdata) / T );
	
	for(t = 1; t <= T+1; t++){

	  /*--- cross validation ---*/

	  if(t == T+1){ mD = mdata; mb = mdata; }
	  else{		
		if(t < T){ vi = range( d*(t-1), (d*t-1) ); }
		else{ vi = range( d*(t-1), (rows(mdata)-1) ); }
		vs = dropc( range(0, rows(mdata)-1), vi );
		mD = mdata[vi][]; mb = mdata[vs][];
	  }// else
	   									 
	  vy0 = mD[][0]; vage0 = mD[][1]; vsex0 = mD[][2]; ms  = mD[][3:];	  
	  vyt = mb[][0]; vage  = mb[][1]; vsex  = mb[][2]; mst = mb[][3:];

	  vmis0_s = ( vsex0 .== 999 ); vmis_s = ( vsex .== 999 );
	  vi0 = vecindex( !vmis0_s ); vi = vecindex( !vmis_s );
	  dm_s = meanc( vsex0[vi0] | vsex[vi] ); dsd_s = sqrt( varc( vsex0[vi0] | vsex[vi] ) );
	  vsex0[vecindex( vmis0_s )] = round(dm_s); vsex[vecindex( vmis_s )] = round(dm_s); 

	  vmis0_a = ( vage0 .== 999 ); vmis_a = ( vage .== 999 );
	  vi0 = vecindex( !vmis0_a ); vi = vecindex( !vmis_a );
	  dm_a = meanc( vage0[vi0] | vage[vi] ); dsd_a = sqrt( varc( vage0[vi0] | vage[vi] ) );
	  vage0[vecindex( vmis0_a )] = round(dm_a); vage[vecindex( vmis_a )] = round(dm_a); 
	  
	  mx0 = (vsex0 - dm_s) ./ dsd_s ~ (vage0 - dm_a) ./ dsd_a;				
	  mx  = (vsex  - dm_s) ./ dsd_s ~ (vage  - dm_a) ./ dsd_a;				
	
	  cn = rows(vy0); ct = rows(vyt); P = columns(mst); Q = columns(mx);	  
	  
	  /*--- start loop for factors ---*/

	  for(K = 1; K <= K_s; K++){	
	
	    if(t == T+1){vh = meanc(mstore); K = sumr( ( vh .== min(vh) ) .* range(1, K_s) ); }

		G = K;

	    /*--- Initial values ---*/
		
	    amB = amLam = amC = new array[L];	 
	    for(l = 0 ; l < L ; l++){
	      amB[l] = zeros(Q, P); amLam[l] = zeros(K, P); 
	      amC[l] = new array[Q+1];
		  amC[l][0] = amC[l][1] = amC[l][2] = zeros(G, P);
	    }// l
	  
	    vphi_M = ones(1, P); mphi_B = ones(Q, P); vphi_L = ones(1, P); mphi_C = ones(Q+1, P);
		
		mz = ( mst .> 0 ) - ( mst .<= 0 );
	    mmu = zeros(L, P); meta = rann(ct, K); mgam = rann(ct, G); vpi0 = meanc( vy0 .== range(0, L-1) );
	    msamp = msamp0 = msamp1 = <>;		

		// parameters for p(sex,age|y) 
		mp_sa = zeros(L, 4);
		mh = (vsex .== 0) .* (vage .== 0) ~ (vsex .== 0) .* (vage .== 1)
		   ~ (vsex .== 1) .* (vage .== 0) ~ (vsex .== 1) .* (vage .== 1);
		mh = mh .* !vmis_s .* !vmis_a;
		for(l = 0 ; l < L ; l++)
	      mp_sa[l][] = sumc( (vyt .== l) .* mh ) ./ sumc( ( vyt .== l ) .* !vmis_s .* !vmis_a );
		
		/*--- MCMC sampling ---*/
	
	    println("\n\nIteration:");

	    /*----------- MCMC: Sampling Start -----------*/
		
	    for(kk = -nburn ; kk < th*nsim ; kk++){  

        /*--- sampling B, Lambda, C ---*/	   

		mh = 1 ~ mx ~ meta ~ mgam ~ ( mx[][0] .* mgam ) ~ (mx[][1] .* mgam);
		for(l = 0 ; l < L ; l++){
		  for(j = 0 ; j < P ; j++){
		    vi = vecindex( ( vyt .== l ) .* ( mst[][j] .!= 999 ) );
		    if( rows(vi) > 0 ){
			  vh = vphi_M[j] | mphi_B[][j] | ( vphi_L[j] .* ones(K, 1) )
			     | ( mphi_C[0][j] .* ones(K, 1) ) | ( mphi_C[1][j] .* ones(K, 1) ) | ( mphi_C[2][j] .* ones(K, 1) );
				 
			  mSig = invert( diag( vh ) + mh[vi][]' mh[vi][] );
		      vmu = mSig * ( mh[vi][]' mz[vi][j] );
		      vl = vmu + choleski(mSig) * rann(1+Q+4*K, 1);

			  mmu[l][j] = vl[0]; amB[l][][j] = vl[1:Q]; amLam[l][][j] = vl[(Q+1):(Q+K)];
			  amC[l][0][][j] = vl[(Q+K+1):(Q+2*K)];	amC[l][1][][j] = vl[(Q+2*K+1):(Q+3*K)];
			  amC[l][2][][j] = vl[(Q+3*K+1):(Q+4*K)];
			}// if
		    else{
			  mmu[l][j] = sqrt( 1 ./ vphi_M[j] ) .* rann(1, 1);
			  amB[l][][j] = sqrt( 1 ./ mphi_B[][j] ) .* rann(Q, 1);
			  amLam[l][][j] = sqrt( 1 ./ vphi_L[j] ) .* rann(K, 1);
			  amC[l][0][][j] = sqrt( 1 ./ mphi_C[0][j] ) .* rann(G, 1);
			  amC[l][1][][j] = sqrt( 1 ./ mphi_C[1][j] ) .* rann(G, 1);
			  amC[l][2][][j] = sqrt( 1 ./ mphi_C[2][j] ) .* rann(G, 1);
			}// else
		  }// j
	    }// l
		
		/*--- sampling eta & gamma ---*/

		for(i = 0 ; i < ct ; i++){
	      vi = vecindex( mst[i][] .!= 999 ); l = vyt[i];
		  mh = amLam[l] | ( amC[l][0] + amC[l][1] .* mx[i][0] + amC[l][2] .* mx[i][1] );	  
	      mSig = invert( unit(2*K) + mh[][vi] * mh[][vi]' ); 
	      vmu = mSig * mh[][vi] * ( mz[i][vi] - mmu[l][vi] - mx[i][] * amB[l][][vi] )';
		  vh = vmu + choleski(mSig) * rann(2*K, 1);
		  meta[i][] = vh[:K-1]'; mgam[i][] = vh[K:]';
	    }// i

	    /*--- sampling vphi_M (phi_B) ---*/
	
	    vh = sumsqrc( mmu );
	    vphi_M = rangamma( 1, P, 0.5 * (1 + L), 0.5 * (1 + vh) );			 	

	    /*--- sampling mphi_B (phi_B) ---*/

	    mh = zeros(Q, P);
	    for(l = 0 ; l < L ; l++){ mh += amB[l].^2; }
		mphi_B[0][] = rangamma( 1, P, 0.5 * (1 + L), 0.5 * (1 + mh[0][]) );			 	
		mphi_B[1][] = rangamma( 1, P, 0.5 * (1 + L), 0.5 * (1 + mh[1][]) );			 	
		
	    /*--- sampling vphi_L (phi_Lambda) ---*/

	    mh = zeros(K, P);
	    for(l = 0 ; l < L ; l++){ mh += amLam[l].^2; }	
		vphi_L = rangamma( 1, P, 0.5 * (1 + L*K), 0.5 * (1 + sumc(mh)) );			 	

		/*--- sampling mphi_C (phi_C) ---*/

	    for(q = 0 ; q <= Q ; q++){
	      mh = zeros(G, P);
	      for(l = 0 ; l < L ; l++){ mh += amC[l][q].^2; }	
		  mphi_C[q][] = rangamma( 1, P, 0.5 * (1 + L*G), 0.5 * (1 + sumc(mh)) );		 	
	    }// q

		/*--- sampling z ---*/	
	
		mh_s = zeros(ct, P);
		for(l = 0 ; l < L ; l++){
		  vi = vecindex( vyt .== l );
		  mh_s[vi][] = mmu[l][] + mx[vi][] * amB[l] + meta[vi][] * amLam[l] + mgam[vi][] * amC[l][0]
		             + ( mx[vi][0] .* mgam[vi][] ) * amC[l][1] + ( mx[vi][1] .* mgam[vi][] ) * amC[l][2];
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

	    /*--- sampling p(sex, age | y) ---*/

		mh = (vsex .== 0) .* (vage .== 0) ~ (vsex .== 0) .* (vage .== 1)
		   ~ (vsex .== 1) .* (vage .== 0) ~ (vsex .== 1) .* (vage .== 1);
		for(l = 0 ; l < L ; l++){
		  vh = randirichlet(1, sumc( (vyt .== l) .* mh ) + 1);
		  mp_sa[l][] = vh ~ ( 1 - sumr(vh) );
		}  

	    /*--- impute sex ---*/			

		 vi = vecindex( vmis_s );
		 if( rows(vi) > 0 ){		 
		   for(j = 0 ; j < rows(vi) ; j++){	i = vi[j]; l = vyt[i]; vh0 = zeros(2, 1);
			 for(s = 0 ; s <= 1 ; s++){			 
			   vx = (s - dm_s) ./ dsd_s ~ (vage[i] - dm_a) ./ dsd_a;				
		       vh = mmu[l][] + vx * amB[l] + meta[i][] * amLam[l] + mgam[i][] * amC[l][0]
			      + ( vx[0] .* mgam[i][] ) * amC[l][1] + ( vx[1] .* mgam[i][] ) * amC[l][2];
		       vp = probn( 0 - vh ); 
		       vp += ( (log(vp) .== -.Inf) - (log(1 - vp) .== -.Inf) ) .* 10^(-10);
			   vd = ( mst[i][] .== 0 ) .* log(vp) + ( mst[i][] .== 1 ) .* log(1 - vp);
			   vh0[s] = mp_sa[l][vage[i]] .* exp( sumr( vd ) );
			 }// s
	         vsex[i] = sumr( cumulate( ( vh0 ./ sumc(vh0) ) )' .< ranu(1, 1) );	
	         mx[i][0] = (vsex[i] - dm_s) ./ dsd_s;							 
		   }// j
		 }// if

	    /*--- impute age ---*/			

		 vi = vecindex( vmis_a );
		 if( rows(vi) > 0 ){		 
		   for(j = 0 ; j < rows(vi) ; j++){	i = vi[j]; l = vyt[i]; vh0 = zeros(2, 1);
			 for(a = 0 ; a <= 1 ; a++){			 
		       vx  = (vsex[i] - dm_s) ./ dsd_s ~ (a - dm_a) ./ dsd_a;				
		       vh = mmu[l][] + vx * amB[l] + meta[i][] * amLam[l] + mgam[i][] * amC[l][0]
			      + ( vx[0] .* mgam[i][] ) * amC[l][1] + ( vx[1] .* mgam[i][] ) * amC[l][2];
		       vp = probn( 0 - vh ); 
		       vp += ( (log(vp) .== -.Inf) - (log(1 - vp) .== -.Inf) ) .* 10^(-10);
			   vd = ( mst[i][] .== 0 ) .* log(vp) + ( mst[i][] .== 1 ) .* log(1 - vp);
			   vh0[a] = mp_sa[l][2*vsex[i]] .* exp( sumr( vd ) );
			 }// a
	         vage[i] = sumr( cumulate( ( vh0 ./ sumc(vh0) ) )' .< ranu(1, 1) );	
	         mx[i][1] = (vage[i]  - dm_a) ./ dsd_a;							 
		   }// j
		 }// if
			
         /*--- sampling pi_y ---*/

	     vd = randirichlet(1, 1 + sumc( vyt .== range(0, L-1) ));
	     vpi_y = vd ~ (1 - sumr(vd)); 

		/*--- storing sample ---*/

	    if(kk >= 0 && !imod(kk, th)){										 

        if(t <= T){

		  /*--- estimate distribution in a target site ---*/

	      mh = zeros(cn, L); mh0 = rann(R, K); mh1 = rann(R, G); vh0 = zeros(cn, 1);
	      for(l = 0; l < L; l++){
		    md0 = mmu[l][] + mh0 * amLam[l] + mh1 * amC[l][0];
	        for(s = 0; s <= 1; s++){
	          for(a = 0; a <= 1; a++){
			    vi = vecindex( ( vsex0 .== s ) .* ( vage0 .== a ) );
			    if( rows( vi ) > 0){
				  vx = (s - dm_s) ./ dsd_s ~ (a - dm_a) ./ dsd_a;				
				  mL = md0 + vx * amB[l] + ( vx[0] .* mh1 ) * amC[l][1] + ( vx[1] .* mh1 ) * amC[l][2];
		          mP = probn( 0 - mL ); 
		          mP += ( (log(mP) .== -.Inf) - (log(1 - mP) .== -.Inf) ) .* 10^(-10);
	              mG = ( ms[vi][] .== 0 ) * log(mP)' + ( ms[vi][] .== 1 ) * log(1 - mP)';	  
		          vh0[vi] = meanr( exp(mG) );
			    }// if
			  }// a
			}// s
			vh = sumr( mp_sa[l][] .* ( (vsex0 .== 0) .* (vage0 .== 0) ~ (vsex0 .== 0) .* (vage0 .== 1)
		                             ~ (vsex0 .== 1) .* (vage0 .== 0) ~ (vsex0 .== 1) .* (vage0 .== 1) ) );		  
	        mh[][l] = vh0 .* vh;
	      }// l
		  mh = mh .* vpi_y;
	      vy = sumr( cumulate( ( mh ./ sumr(mh) )' )' .< ranu(cn, 1) );	

		  /// missing in sex ///
		  if( any( vmis0_s ) ){
		    vi = vecindex( vmis0_s );
	        mh = zeros(rows(vi), L); mh0 = rann(R, K); mh1 = rann(R, G); vh0 = zeros(rows(vi), 1);
	        for(l = 0; l < L; l++){
		      md0 = mmu[l][] + mh0 * amLam[l] + mh1 * amC[l][0];
		      for(j = 0 ; j < rows(vi) ; j++){ i = vi[j];
	            for(s = 0; s <= 1; s++){
				  mx0[i][0] = (s - dm_s) ./ dsd_s;				
				  mL = md0 + mx0[i][] * amB[l] + ( mx0[i][0] .* mh1 ) * amC[l][1] + ( mx0[i][1] .* mh1 ) * amC[l][2];
		          mP = probn( 0 - mL ); 
		          mP += ( (log(mP) .== -.Inf) - (log(1 - mP) .== -.Inf) ) .* 10^(-10);
			      md1 = ( ms[i][] .== 0 ) .* log(mP) + ( ms[i][] .== 1 ) .* log(1 - mP);
			      mh[j][l] += mp_sa[l][vage0[i]+2*s] * meanc( exp( sumr( md1 ) ) );
				}// s
			  }// j		  
	        }// l
			mh = mh .* vpi_y;
	        vy[vi] = sumr( cumulate( ( mh ./ sumr(mh) )' )' .< ranu(rows(vi), 1) );	
		  }// any missing in sex

		  /// missing in age ///
		  if( any( vmis0_a ) ){
		    vi = vecindex( vmis0_a );
	        mh = zeros(rows(vi), L); mh0 = rann(R, K); mh1 = rann(R, G); vh0 = zeros(rows(vi), 1);
	        for(l = 0; l < L; l++){
		      md0 = mmu[l][] + mh0 * amLam[l] + mh1 * amC[l][0];
		      for(j = 0 ; j < rows(vi) ; j++){ i = vi[j];
	            for(a = 0; a <= 1; a++){
	              mx0[i][1] = (a - dm_a) ./ dsd_a;				
				  mL = md0 + mx0[i][] * amB[l] + ( mx0[i][0] .* mh1 ) * amC[l][1]	+ ( mx0[i][1] .* mh1 ) * amC[l][2];
		          mP = probn( 0 - mL ); 
		          mP += ( (log(mP) .== -.Inf) - (log(1 - mP) .== -.Inf) ) .* 10^(-10);
			      md1 = ( ms[i][] .== 0 ) .* log(mP) + ( ms[i][] .== 1 ) .* log(1 - mP);
			      mh[j][l] += mp_sa[l][2*vsex0[i]+a] * meanc( exp( sumr( md1 ) ) );
				}// a
			  }// j		  
	        }// l
			mh = mh .* vpi_y;
	        vy[vi] = sumr( cumulate( ( mh ./ sumr(mh) )' )' .< ranu(rows(vi), 1) );	
		  }// any missing in age
		  
		  msamp |= meanc( vy .== range(0, L-1) ); 

		}else{
		  
          /*--- estimate mutual informatin ---*/

		  dHy = - sumr( vpi_y .* log( vpi_y ) );  // H(Y)
		  
		  vmi = zeros(1, P+2);
		  mh = ( mp_sa[][0] + mp_sa[][1] )' | ( mp_sa[][2] + mp_sa[][3] )';
		  dh = sumc( sumr( ( mh .* vpi_y ) .* log( mh ) ) ) - sumc( sumr( mh .* vpi_y ) .* log( sumr( mh .* vpi_y ) ) );	 // I(Y,sex)
		  vmi[0] = dh / dHy; 
		  mh = ( mp_sa[][0] + mp_sa[][2] )' | ( mp_sa[][1] + mp_sa[][3] )';
		  dh = sumc( sumr( ( mh .* vpi_y ) .* log( mh ) ) ) - sumc( sumr( mh .* vpi_y ) .* log( sumr( mh .* vpi_y ) ) );	 // I(Y,sex)
		  vmi[1] = dh / dHy; 

		  mh0 = rann(R, K); mh1 = rann(R, G); ml0 = ml1 = zeros(2, P);
		  for(l = 0; l < L; l++){
		    md0 = mmu[l][] + mh0 * amLam[l] + mh1 * amC[l][0];
			mh = <>;
	        for(s = 0; s <= 1; s++){
	          for(a = 0; a <= 1; a++){
			    vx = (s - dm_s) ./ dsd_s ~ (a - dm_a) ./ dsd_a;				
			    mL = md0 + vx * amB[l] + ( vx[0] .* mh1 ) * amC[l][1] + ( vx[1] .* mh1 ) * amC[l][2];
		        mP = probn( 0 - mL ); 
		        mP += ( (log(mP) .== -.Inf) - (log(1 - mP) .== -.Inf) ) .* 10^(-10);
			    mh |= meanc( mP ) .* mp_sa[l][2*s+a];
			  }// a
			}// s  
			vh = sumc( mh );
			mh = vh | ( 1 - vh );  // p( x_j | y )

			ml0 += mh .* vpi_y[l] .* log( mh ); ml1 += mh .* vpi_y[l];			
	      }// l  
		  vmi[2:] = ( sumc( ml0 ) - sumc( ml1 .* log( ml1 ) ) ) ./ dHy; 
			
          /*--- estimate conditional mutual information ---*/
					  
		  /*- generate MC sample -*/

	      vy_e = sumr( cumulate( vpi_y' )' .< ranu(R2, 1) );
		  vsex_e = vage_e = zeros(R2, 1); mx_e = zeros(R2, P); mh0 = rann(R2, K); mh1 = rann(R2, G); 
	      for(l = 0; l < L; l++){
			vi = vecindex( vy_e .== l );
			vh = sumr( cumulate( mp_sa[l][]' )' .< ranu(rows(vi), 1) );
			vsex_e[vi] = ( vh .== 2 ) + ( vh .== 3 );
			vage_e[vi] = ( vh .== 1 ) + ( vh .== 3 );
			mh = (vsex_e[vi] - dm_s) ./ dsd_s ~ (vage_e[vi] - dm_a) ./ dsd_a;				
			mL = mmu[l][] + mh0[vi][] * amLam[l] + mh1[vi][] * amC[l][0] + mh * amB[l] 
			   + ( mh[][0] .* mh1[vi][] ) * amC[l][1] + ( mh[][1] .* mh1[vi][] ) * amC[l][2] + rann(rows(vi), P);
		    mx_e[vi][] = ( mL .> 0 ); 
		  }// l

	      /*- compute p(x|y), p(x), p(x_{-j}|y), p(x_{-j}) -*/

		  amP = amG = new array[4];	ml = zeros(R2, 4);
		  mpi0 = mpi1_a = mpi2_a = zeros(R2, 2); mpi1_e = mpi2_e = zeros(R2, P); mh0 = rann(R, K); mh1 = rann(R, G); 
	      for(l = 0; l < L; l++){
		    md0 = mmu[l][] + mh0 * amLam[l] + mh1 * amC[l][0];			
	        for(s = 0; s <= 1; s++){
	          for(a = 0; a <= 1; a++){ q = 2*s + a;
			    vx = (s - dm_s) ./ dsd_s ~ (a - dm_a) ./ dsd_a;				
			    mL = md0 + vx * amB[l] + ( vx[0] .* mh1 ) * amC[l][1] + ( vx[1] .* mh1 ) * amC[l][2];
		        amP[q] = probn( 0 - mL ); 
		        amP[q] += ( (log(amP[q]) .== -.Inf) - (log(1 - amP[q]) .== -.Inf) ) .* 10^(-10);
			    amG[q] = ( mx_e .== 0 ) * log(amP[q])' + ( mx_e .== 1 ) * log(1 - amP[q])';	  
			    ml[][q] = meanr( exp(amG[q]) );
			  }// a
			}// s

			// mpi0: p(x), p(x|y) //
			vh = ( vsex_e .== 0 ) .* ( vage_e .== 0 ) .* ml[][0] .* mp_sa[l][0]
			   + ( vsex_e .== 0 ) .* ( vage_e .== 1 ) .* ml[][1] .* mp_sa[l][1]
			   + ( vsex_e .== 1 ) .* ( vage_e .== 0 ) .* ml[][2] .* mp_sa[l][2]
			   + ( vsex_e .== 1 ) .* ( vage_e .== 1 ) .* ml[][3] .* mp_sa[l][3];
			mpi0[][0] += vh .* vpi_y[l]; mpi0[][1] += vh .* ( vy_e .== l );
			
		    // mpi1_a, mpi2_a: p(x_{-j}) and p(x_{-j}|y) for sex, age //
	        vh = ( vage_e .== 0 ) .* ( ml[][0] .* mp_sa[l][0] + ml[][2] .* mp_sa[l][2] )
			   + ( vage_e .== 1 ) .* ( ml[][1] .* mp_sa[l][1] + ml[][3] .* mp_sa[l][3] );
	        mpi1_a[][0] += vh .* vpi_y[l]; mpi2_a[][0] += vh .* ( vy_e .== l );

	        vh = ( vsex_e .== 0 ) .* ( ml[][0] .* mp_sa[l][0] + ml[][1] .* mp_sa[l][1] )
			   + ( vsex_e .== 1 ) .* ( ml[][2] .* mp_sa[l][2] + ml[][3] .* mp_sa[l][3] );
	        mpi1_a[][1] += vh .* vpi_y[l]; mpi2_a[][1] += vh .* ( vy_e .== l );
			
		    // mpi1_e, mpi2_e: p(x_{-j}), p(x_{-j}|y) //
			vl = zeros(R2, 1);
 	        for(j = 0; j < P; j++){
	          for(s = 0; s <= 1; s++){
	            for(a = 0; a <= 1; a++){ q = 2*s + a;
			      vi = vecindex( ( vsex_e .== s ) .* ( vage_e .== a ) );
			      mh = amG[q][vi][] - ( ( mx_e[vi][j] .== 0 ) .* log( amP[q][][j]' )
				                      + ( mx_e[vi][j] .== 1 ) .* log(1 - amP[q][][j]') );
			      vl[vi] = mp_sa[l][q] .* meanr( exp(mh) );
			  	}// a
			  }// s
			  mpi1_e[][j] += vl .* vpi_y[l]; mpi2_e[][j] += vl .* ( vy_e .== l );
			}// j			
		  }// l
		  ml1 = mpi1_a ~ mpi1_e; ml2 = mpi2_a ~ mpi2_e;
		  vh = meanc( log( mpi0[][1] ) ) - meanc( log( ml2 ) ) + meanc( log( ml1 ) ) - meanc( log( mpi0[][0] ) );		  
		  vcmi = vh ./ dHy;
		  
		  msamp0 |= vmi .* 100; msamp1 |= vcmi .* 100;
		}//else
	    }// saving samples
	
	    /*--- print counter ---*/

	    if(!imod(kk, 1000)){
	      if(t <= T){ println(sprint("cross validation:",t,", # of factor:",K,", MCMC iteration:",kk)); }
	      else{ println(sprint("selected # of factor:",int(K),", MCMC iteration:",kk)); }
	    }	  
										  
	    }//kk [MCMC]		 	

	  if(t <= T){ vpi = meanc( msamp ); mstore[t-1][K-1] = sumr( fabs(vpi0 - vpi) ); }
	  else{ K = K_s; }

	}//K [factor]		 	

	}//t [cross-validation]		 	

	/*------------ S A M P L I N G   E N D --------------*/

	vh0 = msamp0[][0]; vh1 = msamp0[][1]; msamp0[][0] = vh1; msamp0[][1] = vh0;
	vh0 = msamp1[][0]; vh1 = msamp1[][1]; msamp1[][0] = vh1; msamp1[][1] = vh0;

	/*--- output ---*/
		
	savemat("result-mi.csv",  msamp0);
	savemat("result-cmi.csv", msamp1);
	
	println("\n\nTime: ", timespan(time));
}
