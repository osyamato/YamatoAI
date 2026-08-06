import Foundation

struct WeightedValue {

    func calculate(

        probability: Float,

        value: EmbeddingVector

    ) -> EmbeddingVector {

        var weightedValues: [Float] = []

        for index in value.values.indices {

            let weightedValue =

                probability * value.values[index]

            weightedValues.append(weightedValue)

        }

        return EmbeddingVector(

            values: weightedValues

        )

    }

}
