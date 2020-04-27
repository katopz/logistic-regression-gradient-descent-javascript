import * as math from 'mathjs'
import csvToMatrix from 'csv-to-array-matrix'

csvToMatrix('./src/data.csv', init)

function init(matrix: any) {
  // Part 0: Preparation
  console.log('Part 0: Preparation ...\n')

  let X = math.evaluate('matrix[:, 1:2]', {
    matrix
  })
  let y = math.evaluate('matrix[:, 3]', {
    matrix
  })

  let m = y.length
  let n = X[0].length

  // Part 1: Cost Function and Gradient
  console.log('Part 1: Cost Function and Gradient ...\n')

  // Add Intercept Term
  X = math.concat(math.ones([m, 1]).valueOf() as any, X)

  let theta = Array(n + 1)
    .fill(0)
    .map(() => [0])
  let { cost: untrainedThetaCost, grad } = costFunction(theta, X, y)

  console.log('cost: ', untrainedThetaCost)
  console.log('\n')
  console.log('grad: ', grad)
  console.log('\n')

  // Part 2: Gradient Descent (without feature scaling)
  console.log('Part 2: Gradient Descent ...\n')

  const ALPHA = 0.001
  const ITERATIONS = 400

  theta = [[-25], [0], [0]]
  theta = gradientDescent(X, y, theta, ALPHA, ITERATIONS)

  const { cost: trainedThetaCost } = costFunction(theta, X, y)

  console.log('theta: ', theta)
  console.log('\n')
  console.log('cost: ', trainedThetaCost)
  console.log('\n')

  // Part 3: Predict admission of a student with exam scores 45 and 85
  console.log('Part 3: Admission Prediction ...\n')

  let studentVector = [1, 45, 85]
  let prob = sigmoid(
    math.evaluate('studentVector * theta', {
      studentVector,
      theta
    })
  )

  console.log('Predicted admission for student with scores 45 and 85 in exams: ', prob)
}

function sigmoid(z: number) {
  let g = math.evaluate(`1 ./ (1 + e.^-z)`, {
    z
  })

  return g
}

function costFunction(theta: any, X: number, y: any) {
  const m = y.length

  let h = sigmoid(
    math.evaluate(`X * theta`, {
      X,
      theta
    })
  )

  const cost = math.evaluate(`(1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h))`, {
    h,
    y,
    m
  })

  const grad = math.evaluate(`(1 / m) * (h - y)' * X`, {
    h,
    y,
    m,
    X
  })

  return { cost, grad }
}

function gradientDescent(X: number, y: any, theta: any, ALPHA: number, ITERATIONS: number) {
  const m = y.length

  for (let i = 0; i < ITERATIONS; i++) {
    let h = sigmoid(
      math.evaluate(`X * theta`, {
        X,
        theta
      })
    )

    theta = math.evaluate(`theta - ALPHA / m * ((h - y)' * X)'`, {
      theta,
      ALPHA,
      m,
      X,
      y,
      h
    })
  }

  return theta
}
