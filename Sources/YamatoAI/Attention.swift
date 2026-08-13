import Foundation

struct Attention {

    var wq: Matrix

    var wk: Matrix

    var wv: Matrix


    func calculate(

        embeddings: [EmbeddingVector]

    ) -> [EmbeddingVector] {

        var results: [EmbeddingVector] = []

        var queries: [EmbeddingVector] = []

        var keys: [EmbeddingVector] = []

        var values: [EmbeddingVector] = []


        // -------------------------
        // Query・Key・Valueを作る
        // -------------------------

        for embedding in embeddings {

            let query = Query().make(

                from: embedding,

                using: wq

            )

            queries.append(

                query

            )


            let key = Key().make(

                from: embedding,

                using: wk

            )

            keys.append(

                key

            )


            let value = Value().make(

                from: embedding,

                using: wv

            )

            values.append(

                value

            )
        }


        // -------------------------
        // Attention計算
        // -------------------------

        for query in queries {

            var scores: [Float] = []

            let dimension = Float(

                query.values.count

            )


            // -------------------------
            // Query × Key
            // -------------------------

            for key in keys {

                let score = DotProduct().calculate(

                    between: query,

                    and: key

                )

                let scaledScore =
                    score / sqrt(dimension)

                scores.append(

                    scaledScore

                )
            }


            // -------------------------
            // Softmax
            // -------------------------

            let probabilities = Softmax().calculate(

                scores: scores

            )


            // -------------------------
            // Weighted Value
            // -------------------------

            let contextVector = WeightedValue().calculate(

                probabilities: probabilities,

                values: values

            )

            results.append(

                contextVector

            )
        }


        return results
    }


    // MARK: - Update

    mutating func update(

        wqGradient: Matrix,

        wkGradient: Matrix,

        wvGradient: Matrix,

        learningRate: Float

    ) {

        // -------------------------
        // Wq
        // -------------------------

        var newWqValues = wq.values

        for rowIndex in wq.values.indices {

            for columnIndex in wq.values[rowIndex].indices {

                newWqValues[rowIndex][columnIndex] -=
                    learningRate
                    * wqGradient.values[rowIndex][columnIndex]
            }
        }

        wq = Matrix(

            values: newWqValues

        )


        // -------------------------
        // Wk
        // -------------------------

        var newWkValues = wk.values

        for rowIndex in wk.values.indices {

            for columnIndex in wk.values[rowIndex].indices {

                newWkValues[rowIndex][columnIndex] -=
                    learningRate
                    * wkGradient.values[rowIndex][columnIndex]
            }
        }

        wk = Matrix(

            values: newWkValues

        )


        // -------------------------
        // Wv
        // -------------------------

        var newWvValues = wv.values

        for rowIndex in wv.values.indices {

            for columnIndex in wv.values[rowIndex].indices {

                newWvValues[rowIndex][columnIndex] -=
                    learningRate
                    * wvGradient.values[rowIndex][columnIndex]
            }
        }

        wv = Matrix(

            values: newWvValues

        )
    }
}
