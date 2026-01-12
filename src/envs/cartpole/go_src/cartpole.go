package main

import (
	"C"
	"math"
	"math/rand"
	"time"
	"unsafe"
)

// Constants
const (
	Gravity       = 9.8
	MassCart      = 1.0
	MassPole      = 0.1
	TotalMass     = MassCart + MassPole
	Length        = 0.5 // actually half the pole's length
	PoleMassLength = MassPole * Length
	ForceMag      = 10.0
	Tau           = 0.02
	ThetaThreshold = 12.0 * 2.0 * math.Pi / 360.0
	XThreshold    = 2.4
)

// We keep state in a struct, but we will pass it back and forth as a C pointer logic
// or just keep an ID map if we wanted multiple envs.
// For simplicity and speed in this benchmark, we can allocate memory in Go 
// and pass the pointer to Python, but Go's garbage collector might move it unless pinned.
// Better: Python allocates the memory (byte buffer) and passes it to Go to write into.
// Or: We use a simple static struct if single threaded, or heap alloc with manual management.
// Given cgo constraints, let's use C.malloc via cgo or just let Python hold the state array 
// and pass the 4 doubles to Go every time?
// Passing 4 doubles individually is slow. Passing a pointer to an array is fast.
// Let's rely on Python allocating a [4]float64 array and passing the pointer.

// However, we also need internal state like 'steps_beyond_terminated'.
// So we need a struct instance.

type CartPoleState struct {
	X                     float64
	XDot                  float64
	Theta                 float64
	ThetaDot              float64
	StepsBeyondTerminated int
}

//export NewCartPole
func NewCartPole() *C.double {
    // To keep it simple and avoid GC issues with passing Go pointers to C/Python:
    // We will allocate the memory using C's malloc (via C wrapper or just simple Go hack)
    // Actually, safest for c-shared is to just return an ID or use C memory.
    // Let's try a pointer to a Go struct, but we must ensure it's not collected.
    // A common pattern is using a map ID -> *Struct.
    
    // BUT, for raw performance testing, map lookups add overhead.
    // Let's use `C.malloc` if possible? importing "C" allows `C.malloc`.
    // We need to store the 5 fields (4 doubles + 1 int -> aligned to doubles, say 5 doubles).
    
    // Let's keep it simple: We return a pointer to a Go struct, using a handle is safer but slower.
    // We will use the Handle pattern for safety.
    
    id := registerInstance(&CartPoleState{StepsBeyondTerminated: -1})
    return (*C.double)(unsafe.Pointer(uintptr(id))) // Casting ID to pointer to satisfy signature
}

// Global map to store instances
var instances = make(map[int]*CartPoleState)
var nextID = 0

func registerInstance(s *CartPoleState) int {
    instances[nextID] = s
    id := nextID
    nextID++
    return id
}

//export Reset
func Reset(ptr *C.double) {
    id := int(uintptr(unsafe.Pointer(ptr)))
    s := instances[id]
    
    s.X = rand.Float64()*0.1 - 0.05
    s.XDot = rand.Float64()*0.1 - 0.05
    s.Theta = rand.Float64()*0.1 - 0.05
    s.ThetaDot = rand.Float64()*0.1 - 0.05
    s.StepsBeyondTerminated = -1
}

//export GetState
func GetState(ptr *C.double, outPtr *C.double) {
    id := int(uintptr(unsafe.Pointer(ptr)))
    s := instances[id]
    
    // Convert C array pointer to Go slice for easy copying (or just pointer arithmetic)
    // outPtr is float64 array of size 4
    slice := (*[1 << 30]float64)(unsafe.Pointer(outPtr))[:4:4]
    slice[0] = s.X
    slice[1] = s.XDot
    slice[2] = s.Theta
    slice[3] = s.ThetaDot
}

//export Step
func Step(ptr *C.double, action int, outState *C.double, outReward *C.double, outDone *C.int) {
    id := int(uintptr(unsafe.Pointer(ptr)))
    s := instances[id]
    
    force := -ForceMag
    if action == 1 {
        force = ForceMag
    }
    
    costheta := math.Cos(s.Theta)
    sintheta := math.Sin(s.Theta)
    
    temp := (force + PoleMassLength * s.ThetaDot * s.ThetaDot * sintheta) / TotalMass
    thetaacc := (Gravity * sintheta - costheta * temp) / (Length * (4.0/3.0 - MassPole * costheta * costheta / TotalMass))
    xacc := temp - PoleMassLength * thetaacc * costheta / TotalMass
    
    s.X += Tau * s.XDot
    s.XDot += Tau * xacc
    s.Theta += Tau * s.ThetaDot
    s.ThetaDot += Tau * thetaacc
    
    terminated := s.X < -XThreshold || s.X > XThreshold ||
        s.Theta < -ThetaThreshold || s.Theta > ThetaThreshold
        
    reward := 0.0
    if !terminated {
        reward = 1.0
    } else if s.StepsBeyondTerminated == -1 {
        s.StepsBeyondTerminated = 0
        reward = 1.0
    } else {
        s.StepsBeyondTerminated++
        reward = 0.0
    }
    
    // Write outputs
    slice := (*[1 << 30]float64)(unsafe.Pointer(outState))[:4:4]
    slice[0] = s.X
    slice[1] = s.XDot
    slice[2] = s.Theta
    slice[3] = s.ThetaDot
    
    *outReward = C.double(reward)
    
    doneInt := 0
    if terminated {
        doneInt = 1
    }
    *outDone = C.int(doneInt)
}

func main() {
    rand.Seed(time.Now().UnixNano())
}
