import Foundation

struct DotProductGradient {

    func calculate(

        query: EmbeddingVector,

        key: EmbeddingVector,

        scoreGradient: Float

    ) -> (

        queryGradient: EmbeddingVector,

        keyGradient: EmbeddingVector

    ) {

        var queryGradientValues: [Float] = []

        var keyGradientValues: [Float] = []

        for index in query.values.indices {

            let queryGradient =
                key.values[index]
                * scoreGradient

            let keyGradient =
                query.values[index]
                * scoreGradient

            queryGradientValues.append(

                queryGradient

            )

            keyGradientValues.append(

                keyGradient

            )
        }

        return (

            EmbeddingVector(

                values: queryGradientValues

            ),

            EmbeddingVector(

                values: keyGradientValues

            )
        )
    }
}
