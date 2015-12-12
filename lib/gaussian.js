'use strict';

// MODULES //

var toMatrix = require( 'compute-to-matrix' ),
	matrix = require( 'dstructs-matrix' ),
	mean = require( 'compute-mean' ),
	stdev = require( 'compute-stdev' ),
	exp = require( 'compute-exp' ),
	max = require( 'compute-max' ),
	subtract = require( 'compute-subtract' ),
	sum = require( 'compute-sum' ),
	unique = require( 'compute-unique' ),
	isArrayArray = require( 'validate.io-array-array' ),
	isMatrixLike = require( 'validate.io-matrix-like' );


// FUNCTIONS //

var ln = Math.log;


// CONSTANTS //

var PI = Math.PI;


// GAUSSIAN NAIVE BAYES //

/**
* FUNCTION: GaussianFit( x, y, dist )
*	Naive Bayes fitting object constructor for normal distribution.
*
* @constructor
* @param {Matrix} x - design matrix
* @param {Array|Int8Array|Uint8Array|Uint8ClampedArray|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array} y - vector of class memberships
* @returns {GaussianFit} GaussianFit instance
*/
function GaussianFit( x, y ) {
	this.n = x.shape[ 0 ];
	this.p = x.shape[ 1 ];

	this.classes = unique( y );
	this.nclass = this.classes.length;

	this.fitGaussian( x, y );
} // end FUNCTION GaussianFit()

GaussianFit.prototype.score = require( './score.js' );


/**
* METHOD: fitGaussian( x, y )
*	Fit the data under the assumption that p(x_i|c) follows a normal distribution.
*	Assigns prior and conditional probabilities (cprob) of BayesFit instance.
*
* @param {Matrix} x - design matrix
* @param {Array|Int8Array|Uint8Array|Uint8ClampedArray|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array} y - vector of class memberships
* @returns {Void}
*/
GaussianFit.prototype.fitGaussian = function fitGaussian( x, y ) {
	var ids,
		mu, sigma,
		i, j, c,
		nc;

	this.prior = {};
	this.mu = matrix( [ this.p, this.nclass ] );
	this.sigma = matrix( [ this.p, this.nclass ] );
	for ( i = 0; i < this.nclass; i++ ) {
		ids = [];
		c = this.classes[ i ];
		for ( j = 0; j < this.n; j++ ) {
			if ( y[ j ] === c ) {
				ids.push( j );
			}
		}
		nc = ids.length;
		this.prior[ c ] = ln( nc / this.n );
		for ( j = 0; j < this.p; j++ ) {
			mu = mean( x.mget( ids, [j] ) );
			sigma = stdev( x.mget( ids, [j] ) );
			this.mu.set( j, i, mu );
			this.sigma.set( j, i, sigma );
		}
	}
}; // end METHOD fitGaussian()


/**
* METHOD: calcGaussianProb( x, i )
*	Calculate p(X=x,C=i), i.e. the joint probability of observation x and class i.
*
* @param {Array} x - new observation
* @param {Number} i - class indicator
* @returns {Number} res - log probability
*/
GaussianFit.prototype.calcGaussianProb = function calcGaussianProb( x, i ) {
	var c, j,
		res,
		val;

	c = this.classes[ i ];
	res = this.prior[ c ];

	for ( j = 0; j < this.p; j++ ) {
		val = -0.5 * ln( 2 * PI ) * this.sigma.get( j, i ) - 0.5 * ( x[ j ] - this.mu.get( j, i ) ) / this.sigma.get( j, i );
		res += val;
	}

	return res;
}; // end METHOD calcGaussianProb()


/**
* METHOD: predictOne( x )
*	Predict class membership for one new observation.
*
* @param {Array} x - new observation
* @returns {Number|String} predicted class membership
*/
GaussianFit.prototype.predictOne = function predictOne( x ) {
	var i,
		nClasses = this.classes.length,
		logLik = new Array( nClasses ),
		max, argmax,
		val;

	for ( i = 0; i < nClasses; i++ ) {
		logLik[ i ] = this.calcGaussianProb( x, i );
	}
	max = logLik[ 0 ];
	argmax = this.classes[ 0 ];
	for ( i = 0; i < nClasses; i++ ) {
		val = logLik[ i ];
		if ( val > max ) {
			max = val;
			argmax = this.classes[ i ];
		}
	}
	return argmax;
}; // end METHOD predictOne()


/**
* METHOD: predict( x )
*	Predict class membership for new observation(s).
*
* @param {Matrix|Array} x - new observation(s)
* @returns {Array} array of predicted class memberships
*/
GaussianFit.prototype.predict = function predict( x ) {
	var ret,
		i, j,
		logLik,
		nClasses = this.classes.length,
		max, argmax,
		nrow,
		val;

	if ( isArrayArray( x ) ) {
		x = toMatrix( x );
	}

	// Case A: Predictions for multiple obervations:
	if ( isMatrixLike( x ) ) {
		ret = [];
		nrow = x.shape[ 0 ];
		for ( i = 0; i < nrow; i++ ) {
			logLik = new Array( nClasses );
			for ( j = 0; j < nClasses; j++ ) {
				logLik[ j ] =  this.calcGaussianProb( x.mget( [i], null ).data, j );
			}
			max = logLik[ 0 ];
			argmax = this.classes[ 0 ];
			for ( j = 0; j < nClasses; j++ ) {
					val = logLik[ j ];
					if ( val > max ) {
						max = val;
						argmax = this.classes[ j ];
					}
			}
			ret[ i ] = argmax;
		}
		return ret;
	}
	// Case B: Only one new observation:
	return this.predictOne( x );
}; // end METHOD predict()


/**
* METHOD: predictProbs( x )
*	Calculates class membership probabilities.
*
* @param {Matrix|Array} x - design matrix
* @returns {Array} class probabilities
*/
GaussianFit.prototype.predictProbs = function predictProbs( x ) {
	var ret,
		i, j,
		logLik, denom, a,
		nrow;

	if ( isArrayArray( x ) ) {
		x = toMatrix( x );
	}
	// Case A: Predictions for multiple obervations:
	if ( isMatrixLike( x ) ) {
		ret = new Array( nrow );
		nrow = x.shape[ 0 ];
		for ( i = 0; i < nrow; i++ ) {
			logLik = new Array( this.nclass );
			for ( j = 0; j < this.nclass; j++ ) {
				logLik[ j ] =  this.calcGaussianProb( x.mget( [i], null ).data, j );
			}
			a = max( logLik );
			denom = a + ln( sum( exp( subtract( logLik, a ) ) ) );
			logLik = subtract( logLik, denom );
			ret[ i ] = exp( logLik );
		}
		return ret;
	}
	// Case B: Create prediction for a single observation:
	logLik = new Array( this.nclass );
	for ( j = 0; j < this.nclass; j++ ) {
		logLik[ j ] =  this.calcGaussianProb( x, j );
	}
	a = max( logLik );
	denom = a + ln( sum( exp( subtract( logLik, a ) ) ) );
	logLik = subtract( logLik, denom );
	return exp( logLik );
}; // end METHOD predictProbs()

// EXPORTS //

module.exports = GaussianFit;
