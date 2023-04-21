package core

import (
	"fmt"
	"testing"

	. "github.com/onsi/gomega"
)

func TestAdd(t *testing.T) {
	setup(t)
	v1 := Value(3)
	v2 := Value(4)
	res := v1.Add(v2)
	Expect(res.data).To(Equal(7.))
	Expect(res.grad).To(Equal(0.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(1.))
	Expect(v2.grad).To(Equal(1.))

	v1.SetData(6)
	v2.SetData(5)
	res.Forward()
	Expect(res.data).To(Equal(11.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(1.))
	Expect(v2.grad).To(Equal(1.))
}

func TestSub(t *testing.T) {
	setup(t)
	v1 := Value(3)
	v2 := Value(4)
	res := v1.Sub(v2)
	Expect(res.data).To(Equal(-1.))
	Expect(res.grad).To(Equal(0.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(1.))
	Expect(v2.grad).To(Equal(-1.))

	v1.SetData(6)
	v2.SetData(2)
	res.Forward()
	Expect(res.data).To(Equal(4.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(1.))
	Expect(v2.grad).To(Equal(-1.))
}

func TestMult(t *testing.T) {
	setup(t)
	v1 := Value(3)
	v2 := Value(4)
	res := v1.Mult(v2)
	Expect(res.data).To(Equal(12.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(4.))
	Expect(v2.grad).To(Equal(3.))

	v1.SetData(6)
	v2.SetData(5)
	res.Forward()
	Expect(res.data).To(Equal(30.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(5.))
	Expect(v2.grad).To(Equal(6.))
}

func TestDiv(t *testing.T) {
	setup(t)
	v1 := Value(3)
	v2 := Value(4)
	res := v1.Div(v2)
	Expect(res.data).To(Equal(3. / 4.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(1. / 4.))
	Expect(v2.grad).To(Equal(3.))

	v1.SetData(6)
	v2.SetData(5)
	res.Forward()
	Expect(res.data).To(Equal(6. / 5.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(1. / 5.))
	Expect(v2.grad).To(Equal(6.))
}

func TestPow(t *testing.T) {
	setup(t)
	v1 := Value(3)
	res := v1.Pow(2)
	Expect(res.data).To(Equal(9.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(2. * 3.))

	v1.SetData(6)
	res.Forward()
	Expect(res.data).To(Equal(36.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(2. * 6.))
}

func TestReLU(t *testing.T) {
	setup(t)
	v1 := Value(3)
	res := v1.ReLU()
	Expect(res.data).To(Equal(3.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(1.))

	v1.SetData(-6)
	res.Forward()
	Expect(res.data).To(Equal(0.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(0.))
}

func TestLogistic(t *testing.T) {
	setup(t)
	v1 := Value(3)
	res := v1.Logistic()
	Expect(fmt.Sprintf("%.4f", res.data)).To(Equal("0.9526"))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(fmt.Sprintf("%.4f", v1.grad)).To(Equal("0.0452"))

	v1.SetData(-1)
	res.Forward()
	Expect(fmt.Sprintf("%.4f", res.data)).To(Equal("0.2689"))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(fmt.Sprintf("%.4f", v1.grad)).To(Equal("0.1966"))
}

func TestMore(t *testing.T) {
	setup(t)
	v1 := Value(3.)
	v2 := Value(4.)
	v3 := Value(5.)
	v4 := Value(6.)

	res := v1.Mult(v2).Add(v3).Sub(v4)
	Expect(res.data).To(Equal(11.))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(4.))
	Expect(v2.grad).To(Equal(3.))
	Expect(v3.grad).To(Equal(1.))
	Expect(v4.grad).To(Equal(-1.))

	v1.SetData(2.3)
	v2.SetData(.5)
	v3.SetData(3.2)
	v4.SetData(6.4)
	res.Forward()
	Expect(fmt.Sprintf("%.2f", res.data)).To(Equal("-2.05"))

	res.Zero_grad()
	res.Backward()
	Expect(res.grad).To(Equal(1.))
	Expect(v1.grad).To(Equal(.5))
	Expect(v2.grad).To(Equal(2.3))
	Expect(v3.grad).To(Equal(1.))
	Expect(v4.grad).To(Equal(-1.))

	Visualize(res, "../test.png")
}
