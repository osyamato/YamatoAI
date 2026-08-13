import Foundation

struct OutputProjectionGradient {

    func calculate(

        input: EmbeddingVector,

        outputGradient: EmbeddingVector,

        weight: Matrix

    ) -> OutputProjectionGradientResult {

        // -------------------------
        // Input Gradient
        // -------------------------
        //
        // Woを逆向きに通って
        // Concat側へGradientを返す
        // -------------------------

        let inputGradient =
            InputGradient().calculate(

                outputGradient: outputGradient.values,

                weight: weight

            )


        // -------------------------
        // Weight Gradient
        // -------------------------
        //
        // input × outputGradient
        // -------------------------

        var weightGradientValues:

            [[Float]] = Array(

                repeating: Array(

                    repeating: 0,

                    count: outputGradient.values.count

                ),

                count: input.values.count

            )


        for inputIndex in input.values.indices {

            for outputIndex in outputGradient.values.indices {

                weightGradientValues[inputIndex][outputIndex] =

                    input.values[inputIndex]
                    * outputGradient.values[outputIndex]
            }
        }


        let weightGradient = Matrix(

            values: weightGradientValues

        )


        // -------------------------
        // Result
        // -------------------------

        return OutputProjectionGradientResult(

            inputGradient: inputGradient,

            weightGradient: weightGradient

        )
    }
}
