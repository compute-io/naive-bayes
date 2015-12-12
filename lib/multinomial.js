'use strict';

// MODULES //

var toMatrix = require( 'compute-to-matrix' ),
	matrix = require( 'dstructs-matrix' ),
	exp = require( 'compute-exp' ),
	max = require( 'compute-max' ),
	subtract = require( 'compute-subtract' ),
	sum = require( 'compute-sum' ),
	unique = require( 'compute-unique' ),
	isArrayArray = require( 'validate.io-array-array' ),
	isMatrixLike = require( 'validate.io-matrix-like' );


// FUNCTIONS //

var ln = Math.log;


// MULTINOMIAL FITTING //

/**
* FUNCTION: MultinomialFit( x, y, alpha )
*	Naive Bayes fitting object constructor for multinomial distribution.
*
* @constructor
* @param {Matrix} x - design matrix
* @param {Array|Int8Array|Uint8Array|Uint8ClampedArray|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array} y - vector of class memberships
* @param {Number} alpha - Laplace smoothing parameter
* @returns {MultinomialFit} MultinomialFit instance
*/
function MultinomialFit( x, y, alpha ) {
	this.n = x.shape[ 0 ];
	this.p = x.shape[ 1 ];

	this.classes = unique( y );
	this.nclass = this.classes.length;
	this.alpha = alpha;

	this.fitMultinomial( x, y );
} // end FUNCTION MultinomialFit()

MultinomialFit.prototype.score = require( './score.js' );


/**
* METHOD: fitMultinomial( x, y )
*	Fit the data under the assumption that p(x_i|c) follows a multinomial distribution.
*	Assigns prior and conditional probabilities (cprob) of BayesFit instance.
*
* @param {Matrix} x - design matrix
* @param {Array|Int8Array|Uint8Array|Uint8ClampedArray|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array} y - vector of class memberships
* @returns {Void}
*/
MultinomialFit.prototype.fitMultinomial = function fitMultinomial( x, y ) {
	var prior,
		cprob,
		ids,
		counts,
		totalCount,
		i, j, c,
		nc,
		val;

	prior = {};
	cprob = matrix( [ this.p, this.nclass ] );
	for ( i = 0; i < this.nclass; i++ ) {
		ids = [];
		counts = new Int32Array( this.p );
		c = this.classes[ i ];
		for ( j = 0; j < this.n; j++ ) {
			if ( y[ j ] === c ) {
				ids.push( j );
			}
		}
		nc = ids.length;
		prior[ c ] = ln( nc / this.n );
		for ( j = 0; j < this.p; j++ ) {
			counts[ j ] = sum( x.mget( ids, [j] ) );
		}
		totalCount = sum( counts );
		for ( j = 0; j < this.p; j++ ) {
			val = ln( counts[ j ] + this.alpha ) - ln( totalCount + this.p * this.alpha );
			cprob.set( j, i, val );
		}
	}
	this.prior = prior;
	this.cprob = cprob;
}; // end METHOD fitMultinomial()


/**
* METHOD: calcMultinomProb( i, j )
*	description
*
* @param {Array} x - new observation
* @param {Number} i - class indicator
* @returns {Number} j - variable indicator
*/
MultinomialFit.prototype.calcMultinomProb = function calcMultinomProb( x, i, j ) {
	var c,
		res,
		val;

	c = this.classes[ i ];
	res = this.prior[ c ];
	for ( j = 0; j < this.p; j++ ) {
		val = x[ j ] ? x[ j ] * this.cprob.get( j, i ) : 0;
		res += val;
	}
	return res;
}; // end METHOD calcMultinomProb()


/**
* METHOD: predictOne( x )
*	Predict class membership for one new observation.
*
* @param {Array} x - new observation
* @returns {Number|String} predicted class membership
*/
MultinomialFit.prototype.predictOne = function predictOne( x ) {
	var i, j, c,
		nClasses = this.classes.length,
		logLik = new Array( nClasses ),
		max, argmax,
		val;

	for ( i = 0; i < nClasses; i++ ) {
		c = this.classes[ i ];
		logLik[ i ] = this.prior[ c ];
		for ( j = 0; j < this.p; j++ ) {
			val = x[ j ] ? x[ j ] * this.cprob.get( j, i ) : 0;
			logLik[ i ] += val;
		}
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
MultinomialFit.prototype.predict = function predict( x ) {
	var ret,
		i, j, k, c,
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
				c = this.classes[ j ];
				logLik[ j ] = this.prior[ c ];
				for ( k = 0; k < this.p; k++ ) {
					val = x.get( i, k ) ? x.get( i, k ) * this.cprob.get( k, j ) : 0;
					logLik[ j ] += val;
				}
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
MultinomialFit.prototype.predictProbs = function predictProbs( x ) {
	var ret,
		i, j, k, c,
		logLik, denom, a,
		nrow,
		val;

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
				c = this.classes[ j ];
				logLik[ j ] = this.prior[ c ];
				for ( k = 0; k < this.p; k++ ) {
					val = x.get( i, k ) ? x.get( i, k ) * this.cprob.get( k, j ) : 0;
					logLik[ j ] += val;
				}
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
		c = this.classes[ j ];
		logLik[ j ] = this.prior[ c ];
		for ( k = 0; k < this.p; k++ ) {
			val = x[ k ] * this.cprob.get( k, j );
			logLik[ j ] += val;
		}
	}
	a = max( logLik );
	denom = a + ln( sum( exp( subtract( logLik, a ) ) ) );
	logLik = subtract( logLik, denom );
	return exp( logLik );
}; // end METHOD predictProbs()


// EXPORTS //

module.exports = MultinomialFit;
