import Foundation

struct Attention {
    
    func calculate(
        
        embeddings: [EmbeddingVector],
        
        wq: Matrix,
        
        wk: Matrix,
        
        wv: Matrix
        
    ) -> [EmbeddingVector] {
        
        var results: [EmbeddingVector] = []

        var queries: [EmbeddingVector] = []
        var keys: [EmbeddingVector] = []
        var values: [EmbeddingVector] = []
        
        

        for embedding in embeddings {

            let query = Query().make(
                from: embedding,
                using: wq
            )

            queries.append(query)

            let key = Key().make(
                from: embedding,
                using: wk
            )

            keys.append(key)

            let value = Value().make(
                from: embedding,
                using: wv
            )

            values.append(value)
        }
        
        for query in queries {

            var scores: [Float] = []

            for key in keys {

                let score = DotProduct().calculate(

                    between: query,

                    and: key

                )

                scores.append(score)

            }

            let probabilities = Softmax().calculate(

                scores: scores

            )
            
            let contextVector = WeightedValue().calculate(

                probabilities: probabilities,

                values: values

            )

            results.append(contextVector)

        }
        return results
    }
}
