'use strict';

// MODULES //

var toMatrix = require( 'compute-to-matrix' ),
	isArrayArray = require( 'validate.io-array-array' ),
	isArrayLike = require( 'validate.io-array-like' ),
	isMatrixLike = require( 'validate.io-matrix-like' ),
	isNumber = require( 'validate.io-number-primitive' );


// FUNCTIONS //

var GaussianFit = require( './gaussian.js' ),
	MultinomialFit = require( './multinomial.js' );


// NAIVE BAYES //

/**
* FUNCTION: multinomNB( x, y[, opts ] )
*	Fits a multinomial naive Bayes model.
*
* @param {Matrix|Array} x - design matrix
* @param {Array|Int8Array|Uint8Array|Uint8ClampedArray|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array} y - vector of class memberships
* @param {Object} [opts] - function options
* @param {Number} [opts.alpha] - Laplace smoothing parameter
* @returns {MultinomialFit} MultinomialFit instance
*/
function multinomNB( x, y, opts ) {
	var alpha,
		msg,
		fit;

	if ( isArrayArray( x ) ) {
		x = toMatrix( x );
	} else if ( !isMatrixLike( x ) ) {
		msg = 'invalid input argument. The first argument must be a matrix or an array-of-arrays. Value: `' + x + '`';
		throw new TypeError( msg );
	}
	if ( !isArrayLike( y ) ) {
		throw new TypeError( 'invalid input argument. The second argument must be array-like. Value: `' + y + '`' );
	}

	if ( arguments > 2 ) {
		if ( opts.hasOwnProperty( 'alpha' ) ) {
			if ( !isNumber( opts.alpha ) ) {
				throw new TypeError( 'invalid option. Laplace smoothing option must be a number primitive. Option: `' + opts.alpha + '`.' );
			}
		}
	}
	alpha = opts.alpha || 1;
	fit = new MultinomialFit( x, y, alpha );

	return fit;
} // end FUNCTION multinomNB()


/**
* FUNCTION: gaussianNB( x, y[, opts] )
*	Fits a Gaussian naive Bayes model.
*
* @param {Matrix|Array} x - design matrix
* @param {Array|Int8Array|Uint8Array|Uint8ClampedArray|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array} y - vector of class memberships
* @param {Object} [opts] - function options
* @returns {retType} retDescription
*/
function gaussianNB( x, y ) {
	var msg,
		fit;

	if ( isArrayArray( x ) ) {
		x = toMatrix( x );
	} else if ( !isMatrixLike( x ) ) {
		msg = 'invalid input argument. The first argument must be a matrix or an array-of-arrays. Value: `' + x + '`';
		throw new TypeError( msg );
	}
	if ( !isArrayLike( y ) ) {
		throw new TypeError( 'invalid input argument. The second argument must be array-like. Value: `' + y + '`' );
	}

	fit = new GaussianFit( x, y );

	return fit;
} // end FUNCTION gaussianNB()


// EXPORTS //

module.exports = {};
module.exports.multinomial = multinomNB;
module.exports.gaussian = gaussianNB;
