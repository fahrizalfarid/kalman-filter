package kalmanfilter

type SingleStateKalmanFilter[T float32 | float64] struct {
	A                    T //process dynamics
	B                    T //control dynamics
	C                    T //measurement dynamics
	CurrentStateEstimate T //current state estimate
	CurrentProbEstimate  T //current probability of state estimate
	Q                    T //process covariance
	R                    T //measurement covariance
}

func NewSingleStateKalmanFilter[T float32 | float64](A, B, C, x, P, Q, R T) *SingleStateKalmanFilter[T] {
	return &SingleStateKalmanFilter[T]{
		A:                    A,
		B:                    B,
		C:                    C,
		CurrentStateEstimate: x,
		CurrentProbEstimate:  P,
		Q:                    Q,
		R:                    R,
	}
}

func (k *SingleStateKalmanFilter[T]) CurrentState() T {
	return k.CurrentStateEstimate
}

func (k *SingleStateKalmanFilter[T]) Step(controlInput, measurement T) {
	//prediction step
	predictedStateEstimate := k.A*k.CurrentStateEstimate + k.B*controlInput
	predictedProbEstimate := (k.A*k.CurrentProbEstimate)*k.A + k.Q

	//observation step
	innovation := measurement - k.C*predictedStateEstimate
	innovationCovariance := k.C*predictedProbEstimate*k.C + k.R

	//update step
	kalmanGain := predictedProbEstimate * k.C * 1 / T(innovationCovariance)
	k.CurrentStateEstimate = predictedStateEstimate + kalmanGain*innovation

	//eye(n) = nxn identity matrix
	k.CurrentProbEstimate = (1 - kalmanGain*k.C) * predictedProbEstimate
}
