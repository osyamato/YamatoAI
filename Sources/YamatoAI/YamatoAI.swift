import Foundation

struct YamatoAI {

    var dataset: Dataset

    var vocabulary: Vocabulary

    var embedding: Embedding

    init() {

        let dataset = SampleDataset.make()
        
        var vocabulary = Vocabulary()
        
        vocabulary.build(from: dataset)
        
        let embedding = Embedding(

            dimension: 8,

            vocabularySize: vocabulary.count
            )
            
            self.dataset = dataset

            self.vocabulary = vocabulary

            self.embedding = embedding
 

    }
}
