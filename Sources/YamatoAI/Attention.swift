import Foundation

struct Attention {
    
    func calculate(
        
        embedding: EmbeddingVector,
        
        wq: Matrix,
        
        wk: Matrix,
        
        wv: Matrix
        
    ) -> EmbeddingVector {
        
        
        
        let query = Query().make(
            
            from: embedding,
            
            using: wq
            
        )
        let key = Key().make(
            
            from: embedding,
            
            using: wk
            
        )
        let value = Value().make(
            
            from: embedding,
            
            using: wv
            
        )
        
        let score = DotProduct().calculate(

            between: query,

            and: key

        )
        
        
        
        let scores = [

            score

        ]
        
        let probabilities = Softmax().calculate(

            scores: scores

        )
        
        let weightedValue = WeightedValue().calculate(

            probability: probabilities[0],

            value: value

        )
        
        return weightedValue
    }
    
    
    
}

