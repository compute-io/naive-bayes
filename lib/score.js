'use strict';

// MODULES //

var isArrayLike = require( 'validate.io-array-like' );


// SCORE //

/**
* METHOD: score( x, y )
*	Calculates the mean accuracy of the given test data and labels.
*
* @param {Matrix|Array} x - design matrix
* @param {Array|Int8Array|Uint8Array|Uint8ClampedArray|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array} y - vector of class memberships
* @returns {Number} mean accuracy
*/
function score( x, y ) {
	/*jshint validthis: true */
	if ( !isArrayLike( x ) ) {
		throw new TypeError( 'invalid argument. First argument must be a matrix or array of test data. Value: `' + x + '`' );
	}
	if ( !isArrayLike( y ) ) {
		throw new TypeError( 'invalid argument. Second argument must be an array of labels for the test data. Value: `' + y + '`' );
	}

	var yhat = this.predict( x ),
		n = y.length,
		accuracy = 0,
		i;

	for ( i = 0; i < n; i++ ) {
		if ( yhat[ i ] === y[ i ] ) {
			accuracy += 1;
		}
	}
	accuracy /= n;
	return accuracy;
} // end METHOD score()


// EXPORTS //

module.exports = score;
