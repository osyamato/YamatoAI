import Foundation

struct WeightedValue {

    func calculate(

        probabilities: [Float],

        values: [EmbeddingVector]

    ) -> EmbeddingVector {

        var contextValues = Array(

            repeating: Float(0),

            count: values[0].values.count

        )
        
        for valueIndex in values.indices {

            for dimensionIndex in values[valueIndex].values.indices {

                contextValues[dimensionIndex] +=

                    probabilities[valueIndex]

                    *

                    values[valueIndex].values[dimensionIndex]

            }

        }

        return EmbeddingVector(

            values: contextValues

        )

    }

}
