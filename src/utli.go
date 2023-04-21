package core

import (
	"math/rand"
	"time"
)

func mapSlice[T, U any](data []T, f func(elem T, prev T) U) []U {
	res := make([]U, 0, len(data))

	var prev T
	for _, elem := range data {
		res = append(res, f(elem, prev))
		prev = elem
	}

	return res
}

func concat[T any](data ...[]T) []T {
	out := []T{}
	for _, v := range data {
		out = append(out, v...)
	}
	return out
}

var seededRand *rand.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))

func RandValues(length int) []*value {
	out := []*value{}
	for i := 0; i < length; i++ {
		out = append(out, Value(seededRand.Float64())) // init with random value between -1 and 1
	}
	return out
}

func InitValues(length int, initial float64) []*value {
	out := []*value{}
	for i := 0; i < length; i++ {
		out = append(out, Value(initial))
	}
	return out
}
