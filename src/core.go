package core

import (
	"fmt"
	"math"
)

type value struct {
	data      float64
	grad      float64
	children  []*value
	operation string
	_backward func()
	_forward  func()
}

var doNothing = func() {
	// do nothing
}

func Value(v float64) *value {
	return &value{
		data:      v,
		grad:      0,
		children:  []*value{},
		operation: "",
		_backward: doNothing,
		_forward:  doNothing,
	}
}

func makeNext(v float64, c []*value, op string) *value {
	return &value{
		data:      v,
		grad:      0,
		children:  c,
		operation: op,
		_backward: doNothing,
		_forward:  doNothing,
	}
}

func (self *value) GetData() float64 {
	return self.data
}

func (self *value) SetData(v float64) {
	if len(self.children) > 0 {
		panic("cant set data on none leaf nodes")
	}
	self.data = v
}

func (self *value) Add(other *value) *value {
	out := makeNext(
		self.data+other.data,
		[]*value{self, other},
		"+",
	)

	out._backward = func() {
		self.grad += out.grad
		other.grad += out.grad
	}

	out._forward = func() {
		out.data = self.data + other.data
	}

	return out
}

func (self *value) Sub(other *value) *value {
	out := makeNext(
		self.data-other.data,
		[]*value{self, other},
		"-",
	)

	out._backward = func() {
		self.grad += out.grad
		other.grad -= out.grad
	}

	out._forward = func() {
		out.data = self.data - other.data
	}

	return out
}

func (self *value) Mult(other *value) *value {
	out := makeNext(
		self.data*other.data,
		[]*value{self, other},
		"*",
	)

	out._backward = func() {
		self.grad += other.data * out.grad
		other.grad += self.data * out.grad
	}

	out._forward = func() {
		out.data = self.data * other.data
	}

	return out
}

func (self *value) Div(other *value) *value {
	out := makeNext(
		self.data/other.data,
		[]*value{self, other},
		"/",
	)

	out._backward = func() {
		self.grad += (1 / other.data) * out.grad
		other.grad += self.data * out.grad
	}

	out._forward = func() {
		out.data = self.data / other.data
	}

	return out
}

func (self *value) Pow(other float64) *value {
	out := makeNext(
		math.Pow(self.data, other),
		[]*value{self},
		fmt.Sprintf("Pow^(%.2f)", other),
	)

	out._backward = func() {
		self.grad += other * math.Pow(self.data, other-1) * out.grad
	}

	out._forward = func() {
		out.data = math.Pow(self.data, other)
	}

	return out
}

func (self *value) ReLU() *value {
	v := 0.
	if self.data > 0 {
		v = self.data
	}

	out := makeNext(
		v,
		[]*value{self},
		"ReLU",
	)

	out._backward = func() {
		if self.data > 0 {
			self.grad += 1 * out.grad
		} else {
			self.grad += 0 * out.grad
		}
	}

	out._forward = func() {
		out.data = 0
		if self.data > 0 {
			out.data = self.data
		}
	}

	return out
}

func (self *value) Backward() {
	self.grad = 1
	self._backward()

	var backRecurse func(self *value)
	backRecurse = func(s *value) {
		for _, v := range s.children {
			v._backward()
			backRecurse(v)
		}
	}

	backRecurse(self)
}

func (self *value) Forward() {
	var forwardRecurse func(self *value)
	forwardRecurse = func(s *value) {
		for _, v := range s.children {
			forwardRecurse(v)
		}

		s._forward()
	}

	forwardRecurse(self)
}

func (self *value) Zero_grad() {
	var zeroGradRecurse func(self *value)
	zeroGradRecurse = func(s *value) {
		for _, v := range s.children {
			zeroGradRecurse(v)
		}

		s.grad = 0
	}

	zeroGradRecurse(self)
}
