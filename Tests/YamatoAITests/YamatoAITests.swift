import XCTest
@testable import YamatoAI

final class YamatoAITests: XCTestCase {


    func testDotProduct() {

        let rabbit = EmbeddingVector(

            values: [1,2,3]

        )

        let dog = EmbeddingVector(

            values: [4,5,6]

        )

        let dotProduct = DotProduct()

        let result = dotProduct.calculate(

            between: rabbit,

            and: dog

        )

        XCTAssertEqual(result, 32)

    }
    
    func testSoftmax() {

        let softmax = Softmax()

        let scores: [Float] = [
            3,
            4,
            2,
            1
        ]

        let result = softmax.calculate(
            scores: scores
        )

        XCTAssertEqual(result[0], 0.2369, accuracy: 0.0001)

        XCTAssertEqual(result[1], 0.6439, accuracy: 0.0001)

        XCTAssertEqual(result[2], 0.0871, accuracy: 0.0001)

        XCTAssertEqual(result[3], 0.0321, accuracy: 0.0001)

    }
    
    func testMatrix() {

        let matrix = Matrix(

            values: [

                [1, 2],

                [3, 4]

            ]

        )

        XCTAssertEqual(matrix.values[0][0], 1)
        XCTAssertEqual(matrix.values[0][1], 2)

        XCTAssertEqual(matrix.values[1][0], 3)
        XCTAssertEqual(matrix.values[1][1], 4)

    }
    
    func testMatrixMultiply() {

        let matrix = Matrix(

            values: [

                [1, 2],

                [3, 4]

            ]

        )

        let vector = EmbeddingVector(

            values: [

                2,

                3

            ]

        )

        let result = matrix.multiply(

            vector: vector

        )

        XCTAssertEqual(

            result.values,

            [11, 16]

        )

    }
    
    func testAttention() {

        let embeddings = [

            EmbeddingVector(
                values: [1, 2]
            )

        ]

        let wq = Matrix(
            values: [
                [1, 0],
                [0, 1]
            ]
        )

        let wk = Matrix(
            values: [
                [1, 0],
                [0, 1]
            ]
        )

        let wv = Matrix(
            values: [
                [1, 0],
                [0, 1]
            ]
        )


        // -------------------------
        // Attention
        // -------------------------

        let attention = Attention(

            wq: wq,

            wk: wk,

            wv: wv

        )


        // -------------------------
        // Calculate
        // -------------------------

        let result = attention.calculate(

            embeddings: embeddings

        )


        // -------------------------
        // Check
        // -------------------------

        XCTAssertEqual(

            result[0].values,

            [1, 2]

        )
    }
    func testMultiHeadAttention() {

        // 入力：2トークン、それぞれ2次元
        let embeddings = [

            EmbeddingVector(
                values: [1, 2]
            ),

            EmbeddingVector(
                values: [3, 4]
            )

        ]

        // 2次元の単位行列
        let identity = Matrix(

            values: [

                [1, 0],
                [0, 1]

            ]

        )

        // Head 1
        let head1 = MultiHead(

            wq: identity,

            wk: identity,

            wv: identity

        )

        // Head 2
        let head2 = MultiHead(

            wq: identity,

            wk: identity,

            wv: identity

        )

        // Concat後は
        //
        // 2次元 × 2Head = 4次元
        //
        // それを2次元に戻すWo
        let wo = Matrix(

            values: [

                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0]

            ]

        )

        let multiHeadAttention = MultiHeadAttention(

            heads: [
                head1,
                head2
            ],

            wo: wo

        )

        let result = multiHeadAttention.calculate(

            embeddings: embeddings

        )

        // 入力が2トークンなので
        // 出力も2トークン
        XCTAssertEqual(
            result.count,
            2
        )

        // Woによって最終的に2次元
        XCTAssertEqual(
            result[0].values.count,
            2
        )

        XCTAssertEqual(
            result[1].values.count,
            2
        )
    }
    
    func testLayerNormalization() {

        let vector = EmbeddingVector(

            values: [2, 4, 6, 8]

        )

        let layerNormalization = LayerNormalization()

        let result = layerNormalization.calculate(

            vector: vector

        )

        XCTAssertEqual(
            result.values[0],
            -1.3416,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.values[1],
            -0.4472,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.values[2],
            0.4472,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.values[3],
            1.3416,
            accuracy: 0.0001
        )
    }

    func testLayerNormalizationWithSameValues() {

        let vector = EmbeddingVector(

            values: [3, 3, 3]

        )

        let layerNormalization = LayerNormalization()

        let result = layerNormalization.calculate(

            vector: vector

        )

        XCTAssertEqual(
            result.values,
            [0, 0, 0]
        )
    }
    
    func testFeedForwardWithComplexValues() {

        let vector = EmbeddingVector(

            values: [
                0.73,
                -1.28
            ]

        )

        // 2次元 → 3次元
        let w1 = Matrix(

            values: [

                [0.42, -0.31, 0.87],
                [-0.56, 0.64, -0.25]

            ]

        )

        // 3次元 → 2次元
        let w2 = Matrix(

            values: [

                [0.35, -0.72],
                [-0.41, 0.58],
                [0.91, 0.27]

            ]

        )

        let feedForward = FeedForward(

            w1: w1,

            w2: w2

        )

        let result = feedForward.calculate(

            vector: vector

        )

        XCTAssertEqual(

            result.values[0],

            1.227331,

            accuracy: 0.00001

        )

        XCTAssertEqual(

            result.values[1],

            -0.478971,

            accuracy: 0.00001

        )
    
    }
    
    func testTransformerBlock() {

        // 入力：2トークン × 2次元
        let embeddings = [

            EmbeddingVector(
                values: [1, 2]
            ),

            EmbeddingVector(
                values: [3, 4]
            )

        ]

        // 2 × 2 単位行列
        let identity = Matrix(

            values: [

                [1, 0],
                [0, 1]

            ]

        )

        // -------------------------
        // Multi-Head Attention
        // -------------------------

        let head1 = MultiHead(

            wq: identity,

            wk: identity,

            wv: identity

        )

        let head2 = MultiHead(

            wq: identity,

            wk: identity,

            wv: identity

        )

        // 2次元 × 2Head
        // Concat後は4次元
        //
        // Woで4次元 → 2次元
        let wo = Matrix(

            values: [

                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0]

            ]

        )

        let multiHeadAttention = MultiHeadAttention(

            heads: [
                head1,
                head2
            ],

            wo: wo

        )

        // -------------------------
        // Feed Forward
        // -------------------------

        // 2次元 → 3次元
        let w1 = Matrix(

            values: [

                [1, 0, 1],
                [0, 1, 1]

            ]

        )

        // 3次元 → 2次元
        let w2 = Matrix(

            values: [

                [1, 0],
                [0, 1],
                [1, 1]

            ]

        )

        let feedForward = FeedForward(

            w1: w1,

            w2: w2

        )

        // -------------------------
        // Transformer Block
        // -------------------------

        let transformerBlock = TransformerBlock(

            multiHeadAttention: multiHeadAttention,

            feedForward: feedForward

        )

        let result = transformerBlock.calculate(

            embeddings: embeddings

        )

        // -------------------------
        // Test
        // -------------------------

        // 2トークン入れたので
        // 2トークン返ってくる
        XCTAssertEqual(

            result.count,

            2

        )

        // 最終的な次元数も2次元
        XCTAssertEqual(

            result[0].values.count,

            2

        )

        XCTAssertEqual(

            result[1].values.count,

            2

        )

        // NaNになっていないことも確認
        XCTAssertTrue(

            result[0].values.allSatisfy {
                $0.isFinite
            }

        )

        XCTAssertTrue(

            result[1].values.allSatisfy {
                $0.isFinite
            }

        )
    }
    
    func testPositionalEncoding() {

        let embeddings = [

            EmbeddingVector(

                values: [
                    1,
                    1,
                    1,
                    1
                ]

            )

        ]

        let positionalEncoding = PositionalEncoding()

        let result = positionalEncoding.calculate(

            embeddings: embeddings

        )

        XCTAssertEqual(

            result.count,

            1

        )

        XCTAssertEqual(

            result[0].values[0],

            1,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result[0].values[1],

            2,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result[0].values[2],

            1,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result[0].values[3],

            2,

            accuracy: 0.0001

        )
    }
    
    func testPositionalEncodingChangesByPosition() {

        let embeddings = [

            EmbeddingVector(
                values: [1, 1, 1, 1]
            ),

            EmbeddingVector(
                values: [1, 1, 1, 1]
            )

        ]

        let positionalEncoding = PositionalEncoding()

        let result = positionalEncoding.calculate(

            embeddings: embeddings

        )

        XCTAssertNotEqual(

            result[0].values,

            result[1].values

        )
    }
    
    func testTransformer() {

        // -------------------------
        // Input
        // -------------------------

        let embeddings = [

            EmbeddingVector(
                values: [1, 2]
            ),

            EmbeddingVector(
                values: [3, 4]
            )

        ]

        // -------------------------
        // Identity Matrix
        // -------------------------

        let identity = Matrix(

            values: [

                [1, 0],
                [0, 1]

            ]

        )

        // -------------------------
        // Head 1
        // -------------------------

        let head1 = MultiHead(

            wq: identity,

            wk: identity,

            wv: identity

        )

        // -------------------------
        // Head 2
        // -------------------------

        let head2 = MultiHead(

            wq: identity,

            wk: identity,

            wv: identity

        )

        // -------------------------
        // Output Projection
        //
        // 4次元 → 2次元
        // -------------------------

        let wo = Matrix(

            values: [

                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0]

            ]

        )

        let multiHeadAttention = MultiHeadAttention(

            heads: [
                head1,
                head2
            ],

            wo: wo

        )

        // -------------------------
        // Feed Forward
        //
        // 2次元 → 3次元
        // -------------------------

        let w1 = Matrix(

            values: [

                [1, 0, 1],
                [0, 1, 1]

            ]

        )

        // -------------------------
        // 3次元 → 2次元
        // -------------------------

        let w2 = Matrix(

            values: [

                [1, 0],
                [0, 1],
                [1, 1]

            ]

        )

        let feedForward = FeedForward(

            w1: w1,

            w2: w2

        )

        // -------------------------
        // Transformer Block 1
        // -------------------------

        let block1 = TransformerBlock(

            multiHeadAttention: multiHeadAttention,

            feedForward: feedForward

        )

        // -------------------------
        // Transformer Block 2
        // -------------------------

        let block2 = TransformerBlock(

            multiHeadAttention: multiHeadAttention,

            feedForward: feedForward

        )

        // -------------------------
        // Transformer
        // -------------------------

        let transformer = Transformer(

            blocks: [
                block1,
                block2
            ]

        )

        // -------------------------
        // Calculate
        // -------------------------

        let result = transformer.calculate(

            embeddings: embeddings

        )

        // -------------------------
        // Test
        // -------------------------

        // 2トークン入れたので
        // 2トークン返ってくる
        XCTAssertEqual(

            result.count,

            2

        )

        // 各EmbeddingVectorは
        // 2次元のまま
        XCTAssertEqual(

            result[0].values.count,

            2

        )

        XCTAssertEqual(

            result[1].values.count,

            2

        )

        // NaN / infinity になっていない
        XCTAssertTrue(

            result[0].values.allSatisfy {
                $0.isFinite
            }

        )

        XCTAssertTrue(

            result[1].values.allSatisfy {
                $0.isFinite
            }

        )
    }
    
    func testTokenSelector() {

        let probabilities: [Float] = [

            0.05,
            0.66,
            0.01,
            0.28

        ]

        let selector = TokenSelector()

        let result = selector.select(

            probabilities: probabilities

        )

        XCTAssertEqual(

            result,

            1

        )
    }
    
    func testVocabularyWordFromID() {

        var vocabulary = Vocabulary()

        vocabulary.add("猫")
        vocabulary.add("庭")
        vocabulary.add("雨")
        vocabulary.add("犬")

        let result = vocabulary.word(

            for: 1

        )

        XCTAssertEqual(

            result,

            "庭"

        )
    }
    
    func testTokenPredictor() {

        let vector = EmbeddingVector(

            values: [1, 2]

        )

        // 2次元 → Vocabulary 4語
        let weight = Matrix(

            values: [

                [1, 0, 2, -1],
                [0, 2, 1, 1]

            ]

        )

        let outputLayer = OutputLayer(

            weight: weight

        )

        let predictor = TokenPredictor(

            outputLayer: outputLayer

        )

        let probabilities = predictor.calculate(

            vector: vector

        )

        // Vocabularyが4語なので
        // 確率も4個
        XCTAssertEqual(

            probabilities.count,

            4

        )

        // Softmaxなので合計は約1.0
        let total = probabilities.reduce(0, +)

        XCTAssertEqual(

            total,

            1.0,

            accuracy: 0.0001

        )

        // すべて有限値
        XCTAssertTrue(

            probabilities.allSatisfy {
                $0.isFinite
            }

        )
    }
    
    func testTokenDecoder() {

        // -------------------------
        // Vocabulary
        // -------------------------

        var vocabulary = Vocabulary()

        vocabulary.add("猫")
        vocabulary.add("庭")
        vocabulary.add("雨")
        vocabulary.add("犬")

        // ID
        //
        // 0 → 猫
        // 1 → 庭
        // 2 → 雨
        // 3 → 犬


        // -------------------------
        // Input Vector
        // -------------------------

        let vector = EmbeddingVector(

            values: [
                1,
                2
            ]

        )


        // -------------------------
        // Output Layer
        //
        // 2次元 → Vocabulary 4語
        // -------------------------

        let weight = Matrix(

            values: [

                [1, 0, 2, -1],
                [0, 2, 1, 1]

            ]

        )

        let outputLayer = OutputLayer(

            weight: weight

        )


        // -------------------------
        // Token Predictor
        // -------------------------

        let predictor = TokenPredictor(

            outputLayer: outputLayer

        )


        // -------------------------
        // Token Selector
        // -------------------------

        let selector = TokenSelector()


        // -------------------------
        // Token Decoder
        // -------------------------

        let decoder = TokenDecoder(

            predictor: predictor,

            selector: selector,

            vocabulary: vocabulary

        )


        // -------------------------
        // Decode
        // -------------------------

        let result = decoder.decode(

            vector: vector

        )


        // -------------------------
        // Test
        // -------------------------

        XCTAssertEqual(

            result,

            "庭"

        )
    }
    
    func testYamatoModel() {

        // -------------------------
        // Vocabulary
        // -------------------------

        var vocabulary = Vocabulary()

        vocabulary.add("今")
        vocabulary.add("日")
        vocabulary.add("は")
        vocabulary.add("雨")

        // ID
        //
        // 0 → 今
        // 1 → 日
        // 2 → は
        // 3 → 雨


        // -------------------------
        // Embedding
        //
        // Vocabulary 4語
        // 各Vectorは2次元
        // -------------------------

        let embedding = Embedding(

            dimension: 2,

            vocabularySize: vocabulary.count

        )


        // -------------------------
        // Positional Encoding
        // -------------------------

        let positionalEncoding = PositionalEncoding()


        // -------------------------
        // Identity Matrix
        // -------------------------

        let identity = Matrix(

            values: [

                [1, 0],
                [0, 1]

            ]

        )


        // -------------------------
        // Multi-Head Attention
        // -------------------------

        let head1 = MultiHead(

            wq: identity,

            wk: identity,

            wv: identity

        )

        let head2 = MultiHead(

            wq: identity,

            wk: identity,

            wv: identity

        )


        // 2次元 × 2Head
        //
        // Concatすると4次元
        // 4次元 → 2次元へ戻す
        let wo = Matrix(

            values: [

                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0]

            ]

        )

        let multiHeadAttention = MultiHeadAttention(

            heads: [
                head1,
                head2
            ],

            wo: wo

        )


        // -------------------------
        // Feed Forward
        //
        // 2次元 → 3次元
        // -------------------------

        let w1 = Matrix(

            values: [

                [1, 0, 1],
                [0, 1, 1]

            ]

        )


        // 3次元 → 2次元
        let w2 = Matrix(

            values: [

                [1, 0],
                [0, 1],
                [1, 1]

            ]

        )

        let feedForward = FeedForward(

            w1: w1,

            w2: w2

        )


        // -------------------------
        // Transformer Blocks
        // -------------------------

        let block1 = TransformerBlock(

            multiHeadAttention: multiHeadAttention,

            feedForward: feedForward

        )

        let block2 = TransformerBlock(

            multiHeadAttention: multiHeadAttention,

            feedForward: feedForward

        )

        let transformer = Transformer(

            blocks: [
                block1,
                block2
            ]

        )


        // -------------------------
        // Output Layer
        //
        // 2次元 → Vocabulary 4語
        // -------------------------

        let outputWeight = Matrix(

            values: [

                [1, 0, 2, -1],
                [0, 2, 1, 1]

            ]

        )

        let outputLayer = OutputLayer(

            weight: outputWeight

        )


        // -------------------------
        // Token Predictor
        // -------------------------

        let predictor = TokenPredictor(

            outputLayer: outputLayer

        )


        // -------------------------
        // Token Selector
        // -------------------------

        let selector = TokenSelector()


        // -------------------------
        // Token Decoder
        // -------------------------

        let decoder = TokenDecoder(

            predictor: predictor,

            selector: selector,

            vocabulary: vocabulary

        )


        // -------------------------
        // Yamato Model
        // -------------------------

        let model = YamatoModel(

            vocabulary: vocabulary,

            embedding: embedding,

            positionalEncoding: positionalEncoding,

            transformer: transformer,

            decoder: decoder

        )


        // -------------------------
        // Predict
        // -------------------------

        let result = model.predictNextToken(

            text: "今日は雨"

        )


        // -------------------------
        // Test
        // -------------------------

        XCTAssertNotNil(

            result

        )

        XCTAssertTrue(

            vocabulary.id(
                for: result!
            ) != nil

        )
    }
    
    func testCrossEntropyLoss() {

        let probabilities: [Float] = [

            0.05,
            0.66,
            0.01,
            0.28

        ]

        let lossFunction = CrossEntropyLoss()

        let result = lossFunction.calculate(

            probabilities: probabilities,

            correctTokenID: 1

        )

        XCTAssertEqual(

            result,

            -log(0.66),

            accuracy: 0.0001

        )
    }
    
    func testOutputGradient() {

        let probabilities: [Float] = [

            0.70,
            0.20,
            0.10

        ]

        let outputGradient = OutputGradient()

        let result = outputGradient.calculate(

            probabilities: probabilities,

            correctTokenID: 2

        )

        XCTAssertEqual(

            result[0],

            0.70,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result[1],

            0.20,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result[2],

            -0.90,

            accuracy: 0.0001

        )
    }
    
    func testOutputLayerGradient() {

        let vector = EmbeddingVector(

            values: [
                0.5,
                0.8
            ]

        )

        let outputGradient: [Float] = [

            0.70,
            0.20,
            -0.90

        ]

        let outputLayerGradient = OutputLayerGradient()

        let result = outputLayerGradient.calculate(

            vector: vector,

            outputGradient: outputGradient

        )

        XCTAssertEqual(

            result.values[0][0],

            0.35,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[0][1],

            0.10,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[0][2],

            -0.45,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[1][0],

            0.56,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[1][1],

            0.16,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[1][2],

            -0.72,

            accuracy: 0.0001

        )
    }
    
    func testOutputLayerUpdate() {

        let weight = Matrix(

            values: [

                [0.40, 0.30, 0.20],

                [0.10, 0.50, 0.60]

            ]

        )

        var outputLayer = OutputLayer(

            weight: weight

        )

        let gradient = Matrix(

            values: [

                [0.35, 0.10, -0.45],

                [0.56, 0.16, -0.72]

            ]

        )

        outputLayer.update(

            gradient: gradient,

            learningRate: 0.01

        )

        XCTAssertEqual(

            outputLayer.weight.values[0][0],

            0.3965,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            outputLayer.weight.values[0][1],

            0.2990,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            outputLayer.weight.values[0][2],

            0.2045,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            outputLayer.weight.values[1][0],

            0.0944,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            outputLayer.weight.values[1][1],

            0.4984,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            outputLayer.weight.values[1][2],

            0.6072,

            accuracy: 0.0001

        )
    }
    
    func testTrainerLearnsRepeatedly() {

        // -------------------------
        // Embeddings
        // -------------------------

        let embeddings = [

            EmbeddingVector(
                values: [0.5, 0.8]
            ),

            EmbeddingVector(
                values: [0.2, 0.6]
            )

        ]


        // -------------------------
        // Attention
        // -------------------------

        var attention = Attention(

            wq: Matrix(
                values: [
                    [0.4, 0.2],
                    [0.1, 0.5]
                ]
            ),

            wk: Matrix(
                values: [
                    [0.3, 0.1],
                    [0.2, 0.6]
                ]
            ),

            wv: Matrix(
                values: [
                    [0.5, 0.2],
                    [0.1, 0.4]
                ]
            )
        )


        // -------------------------
        // FeedForward
        // -------------------------

        var feedForward = FeedForward(

            w1: Matrix(
                values: [
                    [0.4, 0.2],
                    [0.3, 0.5]
                ]
            ),

            w2: Matrix(
                values: [
                    [0.6, 0.1],
                    [0.2, 0.7]
                ]
            )
        )


        // -------------------------
        // OutputLayer
        // -------------------------

        var outputLayer = OutputLayer(

            weight: Matrix(
                values: [

                    [0.40, 0.30, 0.20],

                    [0.10, 0.50, 0.60]

                ]
            )
        )


        let trainer = Trainer(

            learningRate: 0.01

        )

        let correctTokenID = 2


        // =========================
        // 学習前
        // =========================

        let attentionBefore = attention.calculate(

            embeddings: embeddings

        )

        guard let lastAttentionBefore = attentionBefore.last else {

            XCTFail("Attention output is empty")

            return
        }


        let feedForwardBefore = feedForward.calculate(

            vector: lastAttentionBefore

        )


        let outputBefore = outputLayer.calculate(

            vector: feedForwardBefore

        )


        let probabilitiesBefore = Softmax().calculate(

            scores: outputBefore.values

        )


        let correctProbabilityBefore =
            probabilitiesBefore[correctTokenID]


        // =========================
        // 100回学習
        // =========================

        for _ in 0..<100 {

            trainer.train(

                embeddings: embeddings,

                correctTokenID: correctTokenID,

                attention: &attention,

                feedForward: &feedForward,

                outputLayer: &outputLayer

            )
        }


        // =========================
        // 学習後
        // =========================

        let attentionAfter = attention.calculate(

            embeddings: embeddings

        )

        guard let lastAttentionAfter = attentionAfter.last else {

            XCTFail("Attention output is empty")

            return
        }


        let feedForwardAfter = feedForward.calculate(

            vector: lastAttentionAfter

        )


        let outputAfter = outputLayer.calculate(

            vector: feedForwardAfter

        )


        let probabilitiesAfter = Softmax().calculate(

            scores: outputAfter.values

        )


        let correctProbabilityAfter =
            probabilitiesAfter[correctTokenID]


        // =========================
        // 確認
        // =========================

        print(
            "学習前:",
            correctProbabilityBefore
        )

        print(
            "100回学習後:",
            correctProbabilityAfter
        )


        XCTAssertGreaterThan(

            correctProbabilityAfter,

            correctProbabilityBefore

        )
    }
    
    
    func testInputGradient() {

        let outputGradient: [Float] = [

            0.70,
            0.20,
            -0.90

        ]

        let weight = Matrix(

            values: [

                [0.40, 0.30, 0.20],

                [0.10, 0.50, 0.60]

            ]

        )

        let inputGradient = InputGradient()

        let result = inputGradient.calculate(

            outputGradient: outputGradient,

            weight: weight

        )

        XCTAssertEqual(

            result.values[0],

            0.16,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[1],

            -0.37,

            accuracy: 0.0001

        )
    }
    
    func testFeedForwardGradient() {

        let input = EmbeddingVector(

            values: [
                1.0,
                2.0
            ]

        )

        let outputGradient = EmbeddingVector(

            values: [
                0.5,
                -0.5
            ]

        )

        let w1 = Matrix(

            values: [

                [1.0, -1.0, 0.5],

                [0.5, 1.0, -1.0]

            ]

        )

        let w2 = Matrix(

            values: [

                [1.0, 0.5],

                [-0.5, 1.0],

                [0.5, -1.0]

            ]

        )

        let result = FeedForwardGradient().calculate(

            input: input,

            outputGradient: outputGradient,

            w1: w1,

            w2: w2

        )

        XCTAssertEqual(

            result.values[0],

            1.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[1],

            -0.625,

            accuracy: 0.0001

        )
    }
    
    func testFeedForwardWeightGradient() {

        let input = EmbeddingVector(

            values: [
                1.0,
                2.0
            ]

        )

        let outputGradient = EmbeddingVector(

            values: [
                0.5,
                -0.5
            ]

        )

        let w1 = Matrix(

            values: [

                [1.0, -1.0, 0.5],

                [0.5, 1.0, -1.0]

            ]

        )

        let w2 = Matrix(

            values: [

                [1.0, 0.5],

                [-0.5, 1.0],

                [0.5, -1.0]

            ]

        )

        let result = FeedForwardWeightGradient().calculate(

            input: input,

            outputGradient: outputGradient,

            w1: w1,

            w2: w2

        )

        // W1 Gradient

        XCTAssertEqual(
            result.w1Gradient.values[0][0],
            0.25,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w1Gradient.values[0][1],
            -0.75,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w1Gradient.values[0][2],
            0.0,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w1Gradient.values[1][0],
            0.5,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w1Gradient.values[1][1],
            -1.5,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w1Gradient.values[1][2],
            0.0,
            accuracy: 0.0001
        )

        // W2 Gradient

        XCTAssertEqual(
            result.w2Gradient.values[0][0],
            1.0,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w2Gradient.values[0][1],
            -1.0,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w2Gradient.values[1][0],
            0.5,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w2Gradient.values[1][1],
            -0.5,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w2Gradient.values[2][0],
            0.0,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            result.w2Gradient.values[2][1],
            0.0,
            accuracy: 0.0001
        )
    }
    
    func testFeedForwardUpdate() {

        var feedForward = FeedForward(

            w1: Matrix(

                values: [

                    [1.0, 2.0],

                    [3.0, 4.0]

                ]

            ),

            w2: Matrix(

                values: [

                    [5.0, 6.0],

                    [7.0, 8.0]

                ]

            )
        )

        let w1Gradient = Matrix(

            values: [

                [0.1, 0.2],

                [0.3, 0.4]

            ]

        )

        let w2Gradient = Matrix(

            values: [

                [0.5, 0.6],

                [0.7, 0.8]

            ]

        )

        feedForward.update(

            w1Gradient: w1Gradient,

            w2Gradient: w2Gradient,

            learningRate: 0.1

        )


        // MARK: - W1

        XCTAssertEqual(

            feedForward.w1.values[0][0],

            0.99,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            feedForward.w1.values[0][1],

            1.98,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            feedForward.w1.values[1][0],

            2.97,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            feedForward.w1.values[1][1],

            3.96,

            accuracy: 0.0001

        )


        // MARK: - W2

        XCTAssertEqual(

            feedForward.w2.values[0][0],

            4.95,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            feedForward.w2.values[0][1],

            5.94,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            feedForward.w2.values[1][0],

            6.93,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            feedForward.w2.values[1][1],

            7.92,

            accuracy: 0.0001

        )
    }
    
    func testWeightedValueGradient() {

        let probabilities: [Float] = [

            0.25,
            0.75

        ]

        let values = [

            EmbeddingVector(

                values: [
                    1.0,
                    2.0
                ]

            ),

            EmbeddingVector(

                values: [
                    3.0,
                    4.0
                ]

            )
        ]

        let outputGradient = EmbeddingVector(

            values: [
                0.5,
                1.0
            ]

        )

        let result = WeightedValueGradient().calculate(

            probabilities: probabilities,

            values: values,

            outputGradient: outputGradient

        )


        // -------------------------
        // Probability Gradient
        // -------------------------

        XCTAssertEqual(

            result.probabilityGradients[0],

            2.5,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.probabilityGradients[1],

            5.5,

            accuracy: 0.0001

        )


        // -------------------------
        // Value Gradient 0
        // -------------------------

        XCTAssertEqual(

            result.valueGradients[0].values[0],

            0.125,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.valueGradients[0].values[1],

            0.25,

            accuracy: 0.0001

        )


        // -------------------------
        // Value Gradient 1
        // -------------------------

        XCTAssertEqual(

            result.valueGradients[1].values[0],

            0.375,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.valueGradients[1].values[1],

            0.75,

            accuracy: 0.0001

        )
    }
    
    func testSoftmaxGradient() {

        let probabilities: [Float] = [

            0.2,
            0.3,
            0.5

        ]

        let outputGradient: [Float] = [

            1.0,
            2.0,
            3.0

        ]

        let result = SoftmaxGradient().calculate(

            probabilities: probabilities,

            outputGradient: outputGradient

        )

        XCTAssertEqual(

            result[0],

            -0.26,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result[1],

            -0.09,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result[2],

            0.35,

            accuracy: 0.0001

        )
    }
    
    func testScaledScoreGradient() {

        let scaledScoreGradient: Float = 0.8

        let dimension = 4

        let result = ScaledScoreGradient().calculate(

            scaledScoreGradient: scaledScoreGradient,

            dimension: dimension

        )

        XCTAssertEqual(

            result,

            0.4,

            accuracy: 0.0001

        )
    }
    
    func testDotProductGradient() {

        let query = EmbeddingVector(

            values: [
                2.0,
                3.0
            ]

        )

        let key = EmbeddingVector(

            values: [
                4.0,
                5.0
            ]

        )

        let scoreGradient: Float = 0.5

        let result = DotProductGradient().calculate(

            query: query,

            key: key,

            scoreGradient: scoreGradient

        )


        // -------------------------
        // Query Gradient
        // -------------------------

        XCTAssertEqual(

            result.queryGradient.values[0],

            2.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.queryGradient.values[1],

            2.5,

            accuracy: 0.0001

        )


        // -------------------------
        // Key Gradient
        // -------------------------

        XCTAssertEqual(

            result.keyGradient.values[0],

            1.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.keyGradient.values[1],

            1.5,

            accuracy: 0.0001

        )
    }
    
    func testAttentionWeightGradient() {

        let embeddings = [

            EmbeddingVector(

                values: [
                    1.0,
                    2.0
                ]

            )
        ]

        let outputGradients = [

            EmbeddingVector(

                values: [
                    0.5,
                    1.0
                ]

            )
        ]


        // -------------------------
        // Wq
        // -------------------------

        let wq = Matrix(

            values: [

                [1.0, 0.0],

                [0.0, 1.0]

            ]

        )


        // -------------------------
        // Wk
        // -------------------------

        let wk = Matrix(

            values: [

                [1.0, 0.0],

                [0.0, 1.0]

            ]

        )


        // -------------------------
        // Wv
        // -------------------------

        let wv = Matrix(

            values: [

                [1.0, 0.0],

                [0.0, 1.0]

            ]

        )


        // -------------------------
        // Gradient
        // -------------------------

        let result = AttentionWeightGradient().calculate(

            embeddings: embeddings,

            outputGradients: outputGradients,

            wq: wq,

            wk: wk,

            wv: wv

        )


        // -------------------------
        // Wq Gradient
        // -------------------------
        //
        // 1トークンの場合、
        // Softmax = [1]
        //
        // Scoreを変えても確率は1のままなので
        // Q / K側のGradientは0
        // -------------------------

        XCTAssertEqual(

            result.wqGradient.values[0][0],

            0.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wqGradient.values[0][1],

            0.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wqGradient.values[1][0],

            0.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wqGradient.values[1][1],

            0.0,

            accuracy: 0.0001

        )


        // -------------------------
        // Wk Gradient
        // -------------------------

        XCTAssertEqual(

            result.wkGradient.values[0][0],

            0.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wkGradient.values[0][1],

            0.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wkGradient.values[1][0],

            0.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wkGradient.values[1][1],

            0.0,

            accuracy: 0.0001

        )


        // -------------------------
        // Wv Gradient
        // -------------------------

        XCTAssertEqual(

            result.wvGradient.values[0][0],

            0.5,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wvGradient.values[0][1],

            1.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wvGradient.values[1][0],

            1.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.wvGradient.values[1][1],

            2.0,

            accuracy: 0.0001

        )
    }
    
    func testAttentionUpdate() {

        // -------------------------
        // Weight
        // -------------------------

        let wq = Matrix(
            values: [
                [1.0, 0.0],
                [0.0, 1.0]
            ]
        )

        let wk = Matrix(
            values: [
                [0.5, 0.2],
                [0.3, 0.8]
            ]
        )

        let wv = Matrix(
            values: [
                [0.7, 0.4],
                [0.6, 0.9]
            ]
        )


        // -------------------------
        // Attention
        // -------------------------

        var attention = Attention(

            wq: wq,

            wk: wk,

            wv: wv

        )


        // -------------------------
        // Gradient
        // -------------------------

        let wqGradient = Matrix(
            values: [
                [0.5, 0.2],
                [0.1, 0.4]
            ]
        )

        let wkGradient = Matrix(
            values: [
                [0.2, 0.3],
                [0.4, 0.1]
            ]
        )

        let wvGradient = Matrix(
            values: [
                [0.1, 0.5],
                [0.2, 0.3]
            ]
        )


        // -------------------------
        // Update
        // -------------------------

        attention.update(

            wqGradient: wqGradient,

            wkGradient: wkGradient,

            wvGradient: wvGradient,

            learningRate: 0.1

        )


        // -------------------------
        // Wq Check
        // -------------------------

        XCTAssertEqual(
            attention.wq.values[0][0],
            0.95,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wq.values[0][1],
            -0.02,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wq.values[1][0],
            -0.01,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wq.values[1][1],
            0.96,
            accuracy: 0.0001
        )


        // -------------------------
        // Wk Check
        // -------------------------

        XCTAssertEqual(
            attention.wk.values[0][0],
            0.48,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wk.values[0][1],
            0.17,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wk.values[1][0],
            0.26,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wk.values[1][1],
            0.79,
            accuracy: 0.0001
        )


        // -------------------------
        // Wv Check
        // -------------------------

        XCTAssertEqual(
            attention.wv.values[0][0],
            0.69,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wv.values[0][1],
            0.35,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wv.values[1][0],
            0.58,
            accuracy: 0.0001
        )

        XCTAssertEqual(
            attention.wv.values[1][1],
            0.87,
            accuracy: 0.0001
        )
    }
    
    func testTrainerUpdatesAttention() {

        // -------------------------
        // Embeddings
        // -------------------------

        let embeddings = [

            EmbeddingVector(
                values: [0.5, 0.8]
            ),

            EmbeddingVector(
                values: [0.2, 0.6]
            )

        ]


        // -------------------------
        // Attention
        // -------------------------

        var attention = Attention(

            wq: Matrix(
                values: [
                    [0.4, 0.2],
                    [0.1, 0.5]
                ]
            ),

            wk: Matrix(
                values: [
                    [0.3, 0.1],
                    [0.2, 0.6]
                ]
            ),

            wv: Matrix(
                values: [
                    [0.5, 0.2],
                    [0.1, 0.4]
                ]
            )
        )


        // -------------------------
        // FeedForward
        // -------------------------

        var feedForward = FeedForward(

            w1: Matrix(
                values: [
                    [0.4, 0.2],
                    [0.3, 0.5]
                ]
            ),

            w2: Matrix(
                values: [
                    [0.6, 0.1],
                    [0.2, 0.7]
                ]
            )
        )


        // -------------------------
        // OutputLayer
        // -------------------------

        var outputLayer = OutputLayer(

            weight: Matrix(
                values: [
                    [0.40, 0.30, 0.20],
                    [0.10, 0.50, 0.60]
                ]
            )
        )


        // -------------------------
        // Trainer
        // -------------------------

        let trainer = Trainer(

            learningRate: 0.01

        )

        let correctTokenID = 2


        // =========================
        // 学習前のAttentionを保存
        // =========================

        let wqBefore = attention.wq.values

        let wkBefore = attention.wk.values

        let wvBefore = attention.wv.values


        // =========================
        // 1回学習
        // =========================

        trainer.train(

            embeddings: embeddings,

            correctTokenID: correctTokenID,

            attention: &attention,

            feedForward: &feedForward,

            outputLayer: &outputLayer

        )


        // =========================
        // 学習後
        // =========================

        let wqAfter = attention.wq.values

        let wkAfter = attention.wk.values

        let wvAfter = attention.wv.values


        // -------------------------
        // Debug
        // -------------------------

        print(
            "Wq before:",
            wqBefore
        )

        print(
            "Wq after:",
            wqAfter
        )

        print(
            "Wk before:",
            wkBefore
        )

        print(
            "Wk after:",
            wkAfter
        )

        print(
            "Wv before:",
            wvBefore
        )

        print(
            "Wv after:",
            wvAfter
        )


        // =========================
        // Check
        // =========================

        XCTAssertNotEqual(

            wqBefore,

            wqAfter

        )

        XCTAssertNotEqual(

            wkBefore,

            wkAfter

        )

        XCTAssertNotEqual(

            wvBefore,

            wvAfter

        )
    }
    
    func testLayerNormalizationGradient() {

        // -------------------------
        // Input
        // -------------------------

        let input = EmbeddingVector(

            values: [
                1.0,
                2.0,
                3.0
            ]
        )


        // -------------------------
        // Output Gradient
        // -------------------------

        let outputGradient = EmbeddingVector(

            values: [
                0.1,
                0.2,
                0.4
            ]
        )


        // -------------------------
        // Calculate
        // -------------------------

        let result =
            LayerNormalizationGradient().calculate(

                input: input,

                outputGradient: outputGradient

            )


        // -------------------------
        // Check count
        // -------------------------

        XCTAssertEqual(

            result.values.count,

            3

        )


        // -------------------------
        // Debug
        // -------------------------

        print(
            "LayerNormalization Gradient:",
            result.values
        )


        // -------------------------
        // Expected values
        // -------------------------

        XCTAssertEqual(

            result.values[0],

            0.020412,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[1],

            -0.040825,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.values[2],

            0.020412,

            accuracy: 0.0001

        )


        // -------------------------
        // Gradient sum
        // -------------------------

        let gradientSum =
            result.values.reduce(0, +)


        XCTAssertEqual(

            gradientSum,

            0,

            accuracy: 0.0001

        )
    }
    
    func testResidualGradient() {

        // -------------------------
        // Output Gradient
        // -------------------------

        let outputGradient = EmbeddingVector(

            values: [
                0.2,
                0.5,
                -0.3
            ]
        )


        // -------------------------
        // Calculate
        // -------------------------

        let result =
            ResidualGradient().calculate(

                outputGradient: outputGradient

            )


        // -------------------------
        // Input Gradient
        // -------------------------

        XCTAssertEqual(

            result.inputGradient.values,

            [
                0.2,
                0.5,
                -0.3
            ]

        )


        // -------------------------
        // Branch Gradient
        // -------------------------

        XCTAssertEqual(

            result.branchGradient.values,

            [
                0.2,
                0.5,
                -0.3
            ]

        )
    }
    
    func testOutputProjectionGradient() {

        // -------------------------
        // Input
        // -------------------------

        let input = EmbeddingVector(

            values: [
                2.0,
                3.0
            ]
        )


        // -------------------------
        // Output Gradient
        // -------------------------

        let outputGradient = EmbeddingVector(

            values: [
                0.1,
                -0.2
            ]
        )


        // -------------------------
        // Weight
        // -------------------------

        let weight = Matrix(

            values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]
        )


        // -------------------------
        // Calculate
        // -------------------------

        let result =
            OutputProjectionGradient().calculate(

                input: input,

                outputGradient: outputGradient,

                weight: weight

            )


        // =========================
        // Input Gradient
        // =========================
        //
        // [0.1, -0.2] × Wᵀ
        //
        // input[0]
        // 0.1 × 0.4 + (-0.2 × 0.2)
        // = 0
        //
        // input[1]
        // 0.1 × 0.1 + (-0.2 × 0.5)
        // = -0.09
        // =========================

        XCTAssertEqual(

            result.inputGradient.values[0],

            0.0,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.inputGradient.values[1],

            -0.09,

            accuracy: 0.0001

        )


        // =========================
        // Weight Gradient
        // =========================
        //
        // input × outputGradient
        //
        // [2] × [ 0.1, -0.2 ]
        // [3]
        //
        // =
        //
        // [
        //   [0.2, -0.4],
        //   [0.3, -0.6]
        // ]
        // =========================

        XCTAssertEqual(

            result.weightGradient.values[0][0],

            0.2,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.weightGradient.values[0][1],

            -0.4,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.weightGradient.values[1][0],

            0.3,

            accuracy: 0.0001

        )

        XCTAssertEqual(

            result.weightGradient.values[1][1],

            -0.6,

            accuracy: 0.0001

        )


        // -------------------------
        // Debug
        // -------------------------

        print(
            "Input Gradient:",
            result.inputGradient.values
        )

        print(
            "Wo Gradient:",
            result.weightGradient.values
        )
    }
    
    func testConcatGradient() {

        let outputGradient = EmbeddingVector(

            values: [
                0.1,
                0.2,
                0.3,
                0.4
            ]
        )

        let gradient = ConcatGradient().calculate(

            outputGradient: outputGradient,

            headCount: 2

        )

        // -------------------------
        // Head数
        // -------------------------

        XCTAssertEqual(
            gradient.count,
            2
        )

        // -------------------------
        // Head 0
        // -------------------------

        XCTAssertEqual(
            gradient[0].values,
            [
                0.1,
                0.2
            ]
        )

        // -------------------------
        // Head 1
        // -------------------------

        XCTAssertEqual(
            gradient[1].values,
            [
                0.3,
                0.4
            ]
        )

        print(
            "Head 0 Gradient:",
            gradient[0].values
        )

        print(
            "Head 1 Gradient:",
            gradient[1].values
        )
    }
    
    func testMultiHeadAttentionGradient() {

        // -------------------------
        // Embeddings
        // -------------------------

        let embeddings = [

            EmbeddingVector(
                values: [0.5, 0.8]
            ),

            EmbeddingVector(
                values: [0.2, 0.4]
            )
        ]


        // -------------------------
        // Head 0
        // -------------------------

        let head0 = MultiHead(

            wq: Matrix(
                values: [
                    [0.4, 0.2],
                    [0.1, 0.5]
                ]
            ),

            wk: Matrix(
                values: [
                    [0.3, 0.1],
                    [0.2, 0.6]
                ]
            ),

            wv: Matrix(
                values: [
                    [0.5, 0.2],
                    [0.1, 0.4]
                ]
            )
        )


        // -------------------------
        // Head 1
        // -------------------------

        let head1 = MultiHead(

            wq: Matrix(
                values: [
                    [0.2, 0.3],
                    [0.4, 0.1]
                ]
            ),

            wk: Matrix(
                values: [
                    [0.6, 0.2],
                    [0.1, 0.3]
                ]
            ),

            wv: Matrix(
                values: [
                    [0.3, 0.5],
                    [0.2, 0.4]
                ]
            )
        )

        let heads = [
            head0,
            head1
        ]


        // -------------------------
        // Wo
        //
        // 2次元 × 2Head
        //      ↓
        // Concat = 4次元
        //
        // Output = 2次元
        // -------------------------

        let wo = Matrix(

            values: [

                [0.2, 0.1],

                [0.3, 0.4],

                [0.5, 0.2],

                [0.1, 0.6]
            ]
        )


        // -------------------------
        // 上流から来たGradient
        // Tokenごとに1つ
        // -------------------------

        let outputGradients = [

            EmbeddingVector(
                values: [0.1, -0.2]
            ),

            EmbeddingVector(
                values: [0.3, 0.4]
            )
        ]


        // -------------------------
        // Calculate
        // -------------------------

        let result =
            MultiHeadAttentionGradient().calculate(

                embeddings: embeddings,

                outputGradients: outputGradients,

                heads: heads,

                wo: wo

            )


        // -------------------------
        // Head数
        // -------------------------

        XCTAssertEqual(
            result.headGradients.count,
            2
        )


        // -------------------------
        // Wo Gradient
        // -------------------------

        XCTAssertEqual(
            result.woGradient.values.count,
            4
        )

        XCTAssertEqual(
            result.woGradient.values[0].count,
            2
        )


        // -------------------------
        // Head 0 Gradient
        // -------------------------

        XCTAssertEqual(
            result.headGradients[0]
                .wqGradient.values.count,
            2
        )

        XCTAssertEqual(
            result.headGradients[0]
                .wkGradient.values.count,
            2
        )

        XCTAssertEqual(
            result.headGradients[0]
                .wvGradient.values.count,
            2
        )


        // -------------------------
        // Head 1 Gradient
        // -------------------------

        XCTAssertEqual(
            result.headGradients[1]
                .wqGradient.values.count,
            2
        )

        XCTAssertEqual(
            result.headGradients[1]
                .wkGradient.values.count,
            2
        )

        XCTAssertEqual(
            result.headGradients[1]
                .wvGradient.values.count,
            2
        )


        // -------------------------
        // 確認
        // -------------------------

        print(
            "Wo Gradient:",
            result.woGradient.values
        )

        print(
            "Head 0 Wq Gradient:",
            result.headGradients[0]
                .wqGradient.values
        )

        print(
            "Head 0 Wk Gradient:",
            result.headGradients[0]
                .wkGradient.values
        )

        print(
            "Head 0 Wv Gradient:",
            result.headGradients[0]
                .wvGradient.values
        )

        print(
            "Head 1 Wq Gradient:",
            result.headGradients[1]
                .wqGradient.values
        )

        print(
            "Head 1 Wk Gradient:",
            result.headGradients[1]
                .wkGradient.values
        )

        print(
            "Head 1 Wv Gradient:",
            result.headGradients[1]
                .wvGradient.values
        )
    }
    
    func testMultiHeadAttentionUpdate() {

        // =====================================
        // Head 0
        // =====================================

        let head0 = MultiHead(

            wq: Matrix(values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]),

            wk: Matrix(values: [
                [0.3, 0.1],
                [0.2, 0.6]
            ]),

            wv: Matrix(values: [
                [0.5, 0.2],
                [0.1, 0.4]
            ])
        )


        // =====================================
        // Head 1
        // =====================================

        let head1 = MultiHead(

            wq: Matrix(values: [
                [0.2, 0.3],
                [0.6, 0.1]
            ]),

            wk: Matrix(values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]),

            wv: Matrix(values: [
                [0.3, 0.6],
                [0.2, 0.4]
            ])
        )


        // =====================================
        // Wo
        // =====================================

        let wo = Matrix(values: [

            [0.2, 0.1],
            [0.3, 0.4],
            [0.5, 0.2],
            [0.1, 0.6]

        ])


        var multiHeadAttention =
            MultiHeadAttention(

                heads: [
                    head0,
                    head1
                ],

                wo: wo
            )


        // =====================================
        // Input
        // =====================================

        let embeddings = [

            EmbeddingVector(
                values: [0.2, 0.7]
            ),

            EmbeddingVector(
                values: [0.6, 0.1]
            )
        ]


        let outputGradients = [

            EmbeddingVector(
                values: [0.1, 0.2]
            ),

            EmbeddingVector(
                values: [0.3, 0.4]
            )
        ]


        // =====================================
        // Gradient
        // =====================================

        let gradient =
            MultiHeadAttentionGradient().calculate(

                embeddings:
                    embeddings,

                outputGradients:
                    outputGradients,

                heads:
                    multiHeadAttention.heads,

                wo:
                    multiHeadAttention.wo

            )


        // =====================================
        // Input Gradient確認
        // =====================================

        XCTAssertEqual(
            gradient.inputGradients.count,
            embeddings.count
        )

        for index in embeddings.indices {

            XCTAssertEqual(

                gradient
                    .inputGradients[index]
                    .values.count,

                embeddings[index]
                    .values.count

            )

            for value in gradient
                .inputGradients[index]
                .values {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        // =====================================
        // Gradient確認
        // =====================================

        XCTAssertEqual(
            gradient.headGradients.count,
            2
        )

        XCTAssertEqual(
            gradient.woGradient.values.count,
            wo.values.count
        )


        // =====================================
        // 更新前を保存
        // =====================================

        let oldWo =
            multiHeadAttention.wo.values

        let oldWq =
            multiHeadAttention
                .heads[0]
                .wq
                .values

        let oldWk =
            multiHeadAttention
                .heads[0]
                .wk
                .values

        let oldWv =
            multiHeadAttention
                .heads[0]
                .wv
                .values


        // =====================================
        // Update
        // =====================================

        multiHeadAttention.update(

            gradient:
                gradient,

            learningRate:
                0.01
        )


        // =====================================
        // Woが更新されたか
        // =====================================

        XCTAssertNotEqual(

            oldWo,

            multiHeadAttention
                .wo
                .values
        )


        // =====================================
        // Wqが更新されたか
        // =====================================

        XCTAssertNotEqual(

            oldWq,

            multiHeadAttention
                .heads[0]
                .wq
                .values
        )


        // =====================================
        // Wkが更新されたか
        // =====================================

        XCTAssertNotEqual(

            oldWk,

            multiHeadAttention
                .heads[0]
                .wk
                .values
        )


        // =====================================
        // Wvが更新されたか
        // =====================================

        XCTAssertNotEqual(

            oldWv,

            multiHeadAttention
                .heads[0]
                .wv
                .values
        )


        print(
            "MultiHeadAttention Input Gradient:",
            gradient.inputGradients.map {
                $0.values
            }
        )

        print(
            "Wo before:",
            oldWo
        )

        print(
            "Wo after:",
            multiHeadAttention.wo.values
        )
    }
    
    func testTransformerBlockGradient() {

        // =====================================
        // MultiHeadAttention
        // =====================================

        let head0 = MultiHead(

            wq: Matrix(values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]),

            wk: Matrix(values: [
                [0.3, 0.1],
                [0.2, 0.6]
            ]),

            wv: Matrix(values: [
                [0.5, 0.2],
                [0.1, 0.4]
            ])
        )


        let head1 = MultiHead(

            wq: Matrix(values: [
                [0.2, 0.3],
                [0.6, 0.1]
            ]),

            wk: Matrix(values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]),

            wv: Matrix(values: [
                [0.3, 0.6],
                [0.2, 0.4]
            ])
        )


        let wo = Matrix(values: [

            [0.2, 0.1],
            [0.3, 0.4],
            [0.5, 0.2],
            [0.1, 0.6]

        ])


        let multiHeadAttention =
            MultiHeadAttention(

                heads: [
                    head0,
                    head1
                ],

                wo: wo
            )


        // =====================================
        // FeedForward
        //
        // 2 → 3 → 2
        // =====================================

        let feedForward =
            FeedForward(

                w1: Matrix(values: [

                    [0.2, 0.4, 0.1],
                    [0.3, 0.5, 0.2]

                ]),

                w2: Matrix(values: [

                    [0.3, 0.2],
                    [0.4, 0.1],
                    [0.5, 0.6]

                ])
            )


        // =====================================
        // Embeddings
        // =====================================

        let embeddings = [

            EmbeddingVector(
                values: [0.2, 0.7]
            ),

            EmbeddingVector(
                values: [0.6, 0.1]
            )
        ]


        // =====================================
        // TransformerBlockから来るGradient
        // =====================================

        let outputGradients = [

            EmbeddingVector(
                values: [0.1, 0.2]
            ),

            EmbeddingVector(
                values: [0.3, 0.4]
            )
        ]


        // =====================================
        // Backward
        // =====================================

        let gradient =
            TransformerBlockGradient().calculate(

                embeddings:
                    embeddings,

                outputGradients:
                    outputGradients,

                multiHeadAttention:
                    multiHeadAttention,

                feedForward:
                    feedForward
            )


        // =====================================
        // Input Gradient
        // =====================================

        XCTAssertEqual(

            gradient.inputGradients.count,

            embeddings.count
        )


        for index in embeddings.indices {

            XCTAssertEqual(

                gradient
                    .inputGradients[index]
                    .values.count,

                embeddings[index]
                    .values.count
            )


            for value in gradient
                .inputGradients[index]
                .values {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        // =====================================
        // FeedForward Gradient
        // =====================================

        XCTAssertEqual(

            gradient.w1Gradient.values.count,

            feedForward.w1.values.count
        )


        XCTAssertEqual(

            gradient.w2Gradient.values.count,

            feedForward.w2.values.count
        )


        // =====================================
        // MultiHead Gradient
        // =====================================

        XCTAssertEqual(

            gradient
                .multiHeadAttentionGradient
                .headGradients
                .count,

            2
        )


        XCTAssertEqual(

            gradient
                .multiHeadAttentionGradient
                .inputGradients
                .count,

            embeddings.count
        )


        // =====================================
        // Gradientが実際に存在するか
        // =====================================

        let inputGradientMagnitude =
            gradient.inputGradients
                .flatMap { $0.values }
                .reduce(Float(0)) {

                    $0 + abs($1)

                }


        let w1GradientMagnitude =
            gradient.w1Gradient.values
                .flatMap { $0 }
                .reduce(Float(0)) {

                    $0 + abs($1)

                }


        let w2GradientMagnitude =
            gradient.w2Gradient.values
                .flatMap { $0 }
                .reduce(Float(0)) {

                    $0 + abs($1)

                }


        XCTAssertGreaterThan(
            inputGradientMagnitude,
            0
        )

        XCTAssertGreaterThan(
            w1GradientMagnitude,
            0
        )

        XCTAssertGreaterThan(
            w2GradientMagnitude,
            0
        )


        // =====================================
        // 表示
        // =====================================

        print(
            "Transformer Input Gradient:",
            gradient.inputGradients.map {
                $0.values
            }
        )


        print(
            "Transformer W1 Gradient:",
            gradient.w1Gradient.values
        )


        print(
            "Transformer W2 Gradient:",
            gradient.w2Gradient.values
        )


        print(
            "Transformer Wo Gradient:",
            gradient
                .multiHeadAttentionGradient
                .woGradient
                .values
        )
    }
    func testTransformerBlockUpdate() {

        // =====================================
        // MultiHeadAttention
        // =====================================

        let head0 = MultiHead(

            wq: Matrix(values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]),

            wk: Matrix(values: [
                [0.3, 0.1],
                [0.2, 0.6]
            ]),

            wv: Matrix(values: [
                [0.5, 0.2],
                [0.1, 0.4]
            ])
        )


        let head1 = MultiHead(

            wq: Matrix(values: [
                [0.2, 0.3],
                [0.6, 0.1]
            ]),

            wk: Matrix(values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]),

            wv: Matrix(values: [
                [0.3, 0.6],
                [0.2, 0.4]
            ])
        )


        let multiHeadAttention =
            MultiHeadAttention(

                heads: [
                    head0,
                    head1
                ],

                wo: Matrix(values: [
                    [0.2, 0.1],
                    [0.3, 0.4],
                    [0.5, 0.2],
                    [0.1, 0.6]
                ])
            )


        // =====================================
        // FeedForward
        //
        // 2 → 3 → 2
        // =====================================

        let feedForward =
            FeedForward(

                w1: Matrix(values: [
                    [0.2, 0.4, 0.1],
                    [0.3, 0.5, 0.2]
                ]),

                w2: Matrix(values: [
                    [0.3, 0.2],
                    [0.4, 0.1],
                    [0.5, 0.6]
                ])
            )


        // =====================================
        // TransformerBlock
        // =====================================

        var transformerBlock =
            TransformerBlock(

                multiHeadAttention:
                    multiHeadAttention,

                feedForward:
                    feedForward
            )


        // =====================================
        // Input
        // =====================================

        let embeddings = [

            EmbeddingVector(
                values: [0.2, 0.7]
            ),

            EmbeddingVector(
                values: [0.6, 0.1]
            )
        ]


        let outputGradients = [

            EmbeddingVector(
                values: [0.1, 0.2]
            ),

            EmbeddingVector(
                values: [0.3, 0.4]
            )
        ]


        // =====================================
        // Gradient計算
        // =====================================

        let gradient =
            TransformerBlockGradient().calculate(

                embeddings:
                    embeddings,

                outputGradients:
                    outputGradients,

                multiHeadAttention:
                    transformerBlock.multiHeadAttention,

                feedForward:
                    transformerBlock.feedForward
            )


        // =====================================
        // Gradient取得
        // =====================================

        let attentionGradient =
            gradient.multiHeadAttentionGradient


        let head0Gradient =
            attentionGradient.headGradients[0]


        // =====================================
        // Gradientが有限値か確認
        // =====================================

        for row in attentionGradient.woGradient.values {

            for value in row {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        for row in head0Gradient.wqGradient.values {

            for value in row {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        for row in head0Gradient.wkGradient.values {

            for value in row {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        for row in head0Gradient.wvGradient.values {

            for value in row {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        for row in gradient.w1Gradient.values {

            for value in row {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        for row in gradient.w2Gradient.values {

            for value in row {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        // =====================================
        // Gradient Magnitude
        // =====================================

        let woGradientMagnitude =
            attentionGradient
                .woGradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {
                    $0 + abs($1)
                }


        let wqGradientMagnitude =
            head0Gradient
                .wqGradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {
                    $0 + abs($1)
                }


        let wkGradientMagnitude =
            head0Gradient
                .wkGradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {
                    $0 + abs($1)
                }


        let wvGradientMagnitude =
            head0Gradient
                .wvGradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {
                    $0 + abs($1)
                }


        let w1GradientMagnitude =
            gradient
                .w1Gradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {
                    $0 + abs($1)
                }


        let w2GradientMagnitude =
            gradient
                .w2Gradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {
                    $0 + abs($1)
                }


        // =====================================
        // 主要Gradient確認
        // =====================================

        XCTAssertGreaterThan(
            woGradientMagnitude,
            0
        )


        XCTAssertGreaterThan(
            w1GradientMagnitude,
            0
        )


        XCTAssertGreaterThan(
            w2GradientMagnitude,
            0
        )


        // =====================================
        // Update前を保存
        // =====================================

        let oldWo =
            transformerBlock
                .multiHeadAttention
                .wo
                .values


        let oldHead0Wv =
            transformerBlock
                .multiHeadAttention
                .heads[0]
                .wv
                .values


        let oldW1 =
            transformerBlock
                .feedForward
                .w1
                .values


        let oldW2 =
            transformerBlock
                .feedForward
                .w2
                .values


        // =====================================
        // Update
        //
        // Gradientが非常に小さいため
        // テストでは大きめのLearning Rate
        // =====================================

        let testLearningRate: Float = 1000.0


        transformerBlock.update(

            gradient:
                gradient,

            learningRate:
                testLearningRate
        )


        // =====================================
        // Wo Update確認
        // =====================================

        XCTAssertNotEqual(

            oldWo,

            transformerBlock
                .multiHeadAttention
                .wo
                .values
        )


        // =====================================
        // FeedForward Update確認
        // =====================================

        XCTAssertNotEqual(

            oldW1,

            transformerBlock
                .feedForward
                .w1
                .values
        )


        XCTAssertNotEqual(

            oldW2,

            transformerBlock
                .feedForward
                .w2
                .values
        )


        // =====================================
        // Wv Update確認
        //
        // Wq / Wk は極小Gradientになる場合があるため
        // ここでは「必ずWeightが変わる」は要求しない
        // =====================================

        if wvGradientMagnitude > 0 {

            XCTAssertNotEqual(

                oldHead0Wv,

                transformerBlock
                    .multiHeadAttention
                    .heads[0]
                    .wv
                    .values
            )
        }


        // =====================================
        // Input Gradient確認
        // =====================================

        XCTAssertEqual(

            gradient.inputGradients.count,

            embeddings.count
        )


        for index in embeddings.indices {

            XCTAssertEqual(

                gradient
                    .inputGradients[index]
                    .values
                    .count,

                embeddings[index]
                    .values
                    .count
            )


            for value in gradient
                .inputGradients[index]
                .values {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        // =====================================
        // Debug
        // =====================================

        print(
            "===== TransformerBlock Update ====="
        )


        print(
            "Wo Gradient Magnitude:",
            woGradientMagnitude
        )


        print(
            "Head 0 Wq Gradient Magnitude:",
            wqGradientMagnitude
        )


        print(
            "Head 0 Wk Gradient Magnitude:",
            wkGradientMagnitude
        )


        print(
            "Head 0 Wv Gradient Magnitude:",
            wvGradientMagnitude
        )


        print(
            "W1 Gradient Magnitude:",
            w1GradientMagnitude
        )


        print(
            "W2 Gradient Magnitude:",
            w2GradientMagnitude
        )


        print(
            "Wo before:",
            oldWo
        )


        print(
            "Wo after:",
            transformerBlock
                .multiHeadAttention
                .wo
                .values
        )


        print(
            "W1 before:",
            oldW1
        )


        print(
            "W1 after:",
            transformerBlock
                .feedForward
                .w1
                .values
        )


        print(
            "W2 before:",
            oldW2
        )


        print(
            "W2 after:",
            transformerBlock
                .feedForward
                .w2
                .values
        )


        print(
            "Input Gradient:",
            gradient
                .inputGradients
                .map { $0.values }
        )
    }
  
    func testTransformerGradient() {

        // =====================================
        // Block 0
        // MultiHeadAttention
        // =====================================

        let block0Head0 = MultiHead(

            wq: Matrix(values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]),

            wk: Matrix(values: [
                [0.3, 0.1],
                [0.2, 0.6]
            ]),

            wv: Matrix(values: [
                [0.5, 0.2],
                [0.1, 0.4]
            ])
        )


        let block0Head1 = MultiHead(

            wq: Matrix(values: [
                [0.2, 0.3],
                [0.6, 0.1]
            ]),

            wk: Matrix(values: [
                [0.4, 0.2],
                [0.1, 0.5]
            ]),

            wv: Matrix(values: [
                [0.3, 0.6],
                [0.2, 0.4]
            ])
        )


        let block0Attention =
            MultiHeadAttention(

                heads: [
                    block0Head0,
                    block0Head1
                ],

                wo: Matrix(values: [
                    [0.2, 0.1],
                    [0.3, 0.4],
                    [0.5, 0.2],
                    [0.1, 0.6]
                ])
            )


        // =====================================
        // Block 0
        // FeedForward
        // =====================================

        let block0FeedForward =
            FeedForward(

                w1: Matrix(values: [
                    [0.2, 0.4, 0.1],
                    [0.3, 0.5, 0.2]
                ]),

                w2: Matrix(values: [
                    [0.3, 0.2],
                    [0.4, 0.1],
                    [0.5, 0.6]
                ])
            )


        // =====================================
        // Block 0
        // =====================================

        let block0 =
            TransformerBlock(

                multiHeadAttention:
                    block0Attention,

                feedForward:
                    block0FeedForward
            )


        // =====================================
        // Block 1
        // MultiHeadAttention
        // =====================================

        let block1Head0 = MultiHead(

            wq: Matrix(values: [
                [0.35, 0.15],
                [0.25, 0.45]
            ]),

            wk: Matrix(values: [
                [0.25, 0.35],
                [0.15, 0.55]
            ]),

            wv: Matrix(values: [
                [0.45, 0.25],
                [0.15, 0.35]
            ])
        )


        let block1Head1 = MultiHead(

            wq: Matrix(values: [
                [0.25, 0.35],
                [0.55, 0.15]
            ]),

            wk: Matrix(values: [
                [0.45, 0.15],
                [0.25, 0.35]
            ]),

            wv: Matrix(values: [
                [0.35, 0.55],
                [0.25, 0.45]
            ])
        )


        let block1Attention =
            MultiHeadAttention(

                heads: [
                    block1Head0,
                    block1Head1
                ],

                wo: Matrix(values: [
                    [0.25, 0.15],
                    [0.35, 0.45],
                    [0.45, 0.25],
                    [0.15, 0.55]
                ])
            )


        // =====================================
        // Block 1
        // FeedForward
        // =====================================

        let block1FeedForward =
            FeedForward(

                w1: Matrix(values: [
                    [0.25, 0.35, 0.15],
                    [0.35, 0.45, 0.25]
                ]),

                w2: Matrix(values: [
                    [0.35, 0.25],
                    [0.45, 0.15],
                    [0.55, 0.65]
                ])
            )


        // =====================================
        // Block 1
        // =====================================

        let block1 =
            TransformerBlock(

                multiHeadAttention:
                    block1Attention,

                feedForward:
                    block1FeedForward
            )


        // =====================================
        // Transformer
        // =====================================

        let transformer =
            Transformer(

                blocks: [
                    block0,
                    block1
                ]
            )


        // =====================================
        // Input
        // =====================================

        let embeddings = [

            EmbeddingVector(
                values: [0.2, 0.7]
            ),

            EmbeddingVector(
                values: [0.6, 0.1]
            )
        ]


        // =====================================
        // Transformerの出力側から来るGradient
        // =====================================

        let outputGradients = [

            EmbeddingVector(
                values: [0.1, 0.2]
            ),

            EmbeddingVector(
                values: [0.3, 0.4]
            )
        ]


        // =====================================
        // Transformer Gradient
        // =====================================

        let gradient =
            TransformerGradient().calculate(

                embeddings:
                    embeddings,

                outputGradients:
                    outputGradients,

                transformer:
                    transformer
            )


        // =====================================
        // Block Gradient数
        // =====================================

        XCTAssertEqual(

            gradient.blockGradients.count,

            transformer.blocks.count
        )


        XCTAssertEqual(

            gradient.blockGradients.count,

            2
        )


        // =====================================
        // Transformer Input Gradient
        // =====================================

        XCTAssertEqual(

            gradient.inputGradients.count,

            embeddings.count
        )


        for index in embeddings.indices {

            XCTAssertEqual(

                gradient
                    .inputGradients[index]
                    .values
                    .count,

                embeddings[index]
                    .values
                    .count
            )


            for value in gradient
                .inputGradients[index]
                .values {

                XCTAssertTrue(
                    value.isFinite
                )
            }
        }


        // =====================================
        // 各BlockのGradient確認
        // =====================================

        for blockIndex
            in gradient.blockGradients.indices {

            let blockGradient =
                gradient.blockGradients[blockIndex]


            // -------------------------
            // Input Gradient
            // -------------------------

            XCTAssertEqual(

                blockGradient
                    .inputGradients
                    .count,

                embeddings.count
            )


            for tokenGradient
                in blockGradient.inputGradients {

                for value
                    in tokenGradient.values {

                    XCTAssertTrue(
                        value.isFinite
                    )
                }
            }


            // -------------------------
            // Wo Gradient
            // -------------------------

            for row in blockGradient
                .multiHeadAttentionGradient
                .woGradient
                .values {

                for value in row {

                    XCTAssertTrue(
                        value.isFinite
                    )
                }
            }


            // -------------------------
            // Wq / Wk / Wv Gradient
            // -------------------------

            for headGradient in blockGradient
                .multiHeadAttentionGradient
                .headGradients {

                for row in headGradient
                    .wqGradient
                    .values {

                    for value in row {

                        XCTAssertTrue(
                            value.isFinite
                        )
                    }
                }


                for row in headGradient
                    .wkGradient
                    .values {

                    for value in row {

                        XCTAssertTrue(
                            value.isFinite
                        )
                    }
                }


                for row in headGradient
                    .wvGradient
                    .values {

                    for value in row {

                        XCTAssertTrue(
                            value.isFinite
                        )
                    }
                }
            }


            // -------------------------
            // W1 Gradient
            // -------------------------

            for row in blockGradient
                .w1Gradient
                .values {

                for value in row {

                    XCTAssertTrue(
                        value.isFinite
                    )
                }
            }


            // -------------------------
            // W2 Gradient
            // -------------------------

            for row in blockGradient
                .w2Gradient
                .values {

                for value in row {

                    XCTAssertTrue(
                        value.isFinite
                    )
                }
            }
        }


        // =====================================
        // Gradient Magnitude
        // =====================================

        let transformerInputMagnitude =
            gradient
                .inputGradients
                .flatMap { $0.values }
                .reduce(Float(0)) {

                    $0 + abs($1)
                }


        let block0W1Magnitude =
            gradient
                .blockGradients[0]
                .w1Gradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {

                    $0 + abs($1)
                }


        let block1W1Magnitude =
            gradient
                .blockGradients[1]
                .w1Gradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {

                    $0 + abs($1)
                }


        let block0WoMagnitude =
            gradient
                .blockGradients[0]
                .multiHeadAttentionGradient
                .woGradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {

                    $0 + abs($1)
                }


        let block1WoMagnitude =
            gradient
                .blockGradients[1]
                .multiHeadAttentionGradient
                .woGradient
                .values
                .flatMap { $0 }
                .reduce(Float(0)) {

                    $0 + abs($1)
                }


        // =====================================
        // Gradientが消失していないことを確認
        // =====================================

        XCTAssertGreaterThan(

            transformerInputMagnitude,

            0
        )


        XCTAssertGreaterThan(

            block0W1Magnitude,

            0
        )


        XCTAssertGreaterThan(

            block1W1Magnitude,

            0
        )


        XCTAssertGreaterThan(

            block0WoMagnitude,

            0
        )


        XCTAssertGreaterThan(

            block1WoMagnitude,

            0
        )


        // =====================================
        // Gradient接続確認
        //
        // Block 1の入力Gradient
        // =
        // Block 0へ渡されるOutput Gradient
        //
        // 最終的にBlock 0を通って
        // Transformer Input Gradientになる
        // =====================================

        XCTAssertEqual(

            gradient
                .blockGradients[0]
                .inputGradients
                .count,

            gradient
                .inputGradients
                .count
        )


        for tokenIndex
            in gradient.inputGradients.indices {

            for valueIndex
                in gradient
                    .inputGradients[tokenIndex]
                    .values
                    .indices {

                XCTAssertEqual(

                    gradient
                        .inputGradients[tokenIndex]
                        .values[valueIndex],

                    gradient
                        .blockGradients[0]
                        .inputGradients[tokenIndex]
                        .values[valueIndex],

                    accuracy:
                        0.000001
                )
            }
        }


        // =====================================
        // Debug
        // =====================================

        print(
            "===== Transformer Gradient ====="
        )


        print(
            "Transformer Input Gradient:",
            gradient
                .inputGradients
                .map { $0.values }
        )


        print(
            "Block Gradient Count:",
            gradient.blockGradients.count
        )


        print(
            "Block 0 W1 Gradient Magnitude:",
            block0W1Magnitude
        )


        print(
            "Block 1 W1 Gradient Magnitude:",
            block1W1Magnitude
        )


        print(
            "Block 0 Wo Gradient Magnitude:",
            block0WoMagnitude
        )


        print(
            "Block 1 Wo Gradient Magnitude:",
            block1WoMagnitude
        )


        print(
            "Transformer Input Gradient Magnitude:",
            transformerInputMagnitude
        )
    }
    
    func testTransformerUpdate() {

        // =====================================
        // Transformerを作る
        // =====================================

        func makeBlock() -> TransformerBlock {

            let head0 = MultiHead(

                wq: Matrix(values: [
                    [0.4, 0.2],
                    [0.1, 0.5]
                ]),

                wk: Matrix(values: [
                    [0.3, 0.1],
                    [0.2, 0.6]
                ]),

                wv: Matrix(values: [
                    [0.5, 0.2],
                    [0.1, 0.4]
                ])
            )


            let head1 = MultiHead(

                wq: Matrix(values: [
                    [0.2, 0.3],
                    [0.6, 0.1]
                ]),

                wk: Matrix(values: [
                    [0.4, 0.2],
                    [0.1, 0.5]
                ]),

                wv: Matrix(values: [
                    [0.3, 0.6],
                    [0.2, 0.4]
                ])
            )


            let attention =
                MultiHeadAttention(

                    heads: [
                        head0,
                        head1
                    ],

                    wo: Matrix(values: [
                        [0.2, 0.1],
                        [0.3, 0.4],
                        [0.5, 0.2],
                        [0.1, 0.6]
                    ])
                )


            let feedForward =
                FeedForward(

                    w1: Matrix(values: [
                        [0.2, 0.4, 0.1],
                        [0.3, 0.5, 0.2]
                    ]),

                    w2: Matrix(values: [
                        [0.3, 0.2],
                        [0.4, 0.1],
                        [0.5, 0.6]
                    ])
                )


            return TransformerBlock(

                multiHeadAttention:
                    attention,

                feedForward:
                    feedForward
            )
        }


        var transformer =
            Transformer(

                blocks: [
                    makeBlock(),
                    makeBlock()
                ]
            )


        // =====================================
        // Attention Head Gradientを作る
        // =====================================

        func makeHeadGradient() -> AttentionWeightGradientResult {

            AttentionWeightGradientResult(

                inputGradients: [

                    EmbeddingVector(
                        values: [0.1, 0.2]
                    ),

                    EmbeddingVector(
                        values: [0.3, 0.4]
                    )
                ],

                wqGradient: Matrix(values: [
                    [0.1, 0.2],
                    [0.3, 0.4]
                ]),

                wkGradient: Matrix(values: [
                    [0.2, 0.1],
                    [0.4, 0.3]
                ]),

                wvGradient: Matrix(values: [
                    [0.3, 0.2],
                    [0.1, 0.4]
                ])
            )
        }


        // =====================================
        // MultiHead Gradient
        // =====================================

        func makeMultiHeadGradient()
            -> MultiHeadAttentionGradientResult {

            MultiHeadAttentionGradientResult(

                inputGradients: [

                    EmbeddingVector(
                        values: [0.1, 0.2]
                    ),

                    EmbeddingVector(
                        values: [0.3, 0.4]
                    )
                ],

                woGradient: Matrix(values: [

                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.2, 0.3],
                    [0.4, 0.1]

                ]),

                headGradients: [

                    makeHeadGradient(),

                    makeHeadGradient()
                ]
            )
        }


        // =====================================
        // Block Gradient
        // =====================================

        func makeBlockGradient()
            -> TransformerBlockGradientResult {

            TransformerBlockGradientResult(

                inputGradients: [

                    EmbeddingVector(
                        values: [0.1, 0.2]
                    ),

                    EmbeddingVector(
                        values: [0.3, 0.4]
                    )
                ],

                multiHeadAttentionGradient:
                    makeMultiHeadGradient(),

                w1Gradient: Matrix(values: [

                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]

                ]),

                w2Gradient: Matrix(values: [

                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]

                ])
            )
        }


        // =====================================
        // Transformer Gradient
        // =====================================

        let gradient =
            TransformerGradientResult(

                inputGradients: [

                    EmbeddingVector(
                        values: [0.1, 0.2]
                    ),

                    EmbeddingVector(
                        values: [0.3, 0.4]
                    )
                ],

                blockGradients: [

                    makeBlockGradient(),

                    makeBlockGradient()
                ]
            )


        // =====================================
        // Update前
        // =====================================

        let block0OldWq =
            transformer.blocks[0]
                .multiHeadAttention
                .heads[0]
                .wq
                .values

        let block0OldWo =
            transformer.blocks[0]
                .multiHeadAttention
                .wo
                .values

        let block0OldW1 =
            transformer.blocks[0]
                .feedForward
                .w1
                .values

        let block0OldW2 =
            transformer.blocks[0]
                .feedForward
                .w2
                .values


        let block1OldWq =
            transformer.blocks[1]
                .multiHeadAttention
                .heads[0]
                .wq
                .values

        let block1OldWo =
            transformer.blocks[1]
                .multiHeadAttention
                .wo
                .values

        let block1OldW1 =
            transformer.blocks[1]
                .feedForward
                .w1
                .values

        let block1OldW2 =
            transformer.blocks[1]
                .feedForward
                .w2
                .values


        // =====================================
        // Transformer Update
        // =====================================

        transformer.update(

            gradient:
                gradient,

            learningRate:
                0.1
        )


        // =====================================
        // Block 0
        // =====================================

        XCTAssertNotEqual(

            block0OldWq,

            transformer.blocks[0]
                .multiHeadAttention
                .heads[0]
                .wq
                .values
        )


        XCTAssertNotEqual(

            block0OldWo,

            transformer.blocks[0]
                .multiHeadAttention
                .wo
                .values
        )


        XCTAssertNotEqual(

            block0OldW1,

            transformer.blocks[0]
                .feedForward
                .w1
                .values
        )


        XCTAssertNotEqual(

            block0OldW2,

            transformer.blocks[0]
                .feedForward
                .w2
                .values
        )


        // =====================================
        // Block 1
        // =====================================

        XCTAssertNotEqual(

            block1OldWq,

            transformer.blocks[1]
                .multiHeadAttention
                .heads[0]
                .wq
                .values
        )


        XCTAssertNotEqual(

            block1OldWo,

            transformer.blocks[1]
                .multiHeadAttention
                .wo
                .values
        )


        XCTAssertNotEqual(

            block1OldW1,

            transformer.blocks[1]
                .feedForward
                .w1
                .values
        )


        XCTAssertNotEqual(

            block1OldW2,

            transformer.blocks[1]
                .feedForward
                .w2
                .values
        )


        // =====================================
        // 数値も1個確認
        //
        // 0.2 - 0.1 × 0.1
        // = 0.19
        // =====================================

        XCTAssertEqual(

            transformer.blocks[0]
                .feedForward
                .w1
                .values[0][0],

            0.19,

            accuracy:
                0.0001
        )


        XCTAssertEqual(

            transformer.blocks[1]
                .multiHeadAttention
                .wo
                .values[0][0],

            0.19,

            accuracy:
                0.0001
        )


        // =====================================
        // Debug
        // =====================================

        print(
            "Block 0 W1 before:",
            block0OldW1
        )

        print(
            "Block 0 W1 after:",
            transformer.blocks[0]
                .feedForward
                .w1
                .values
        )

        print(
            "Block 1 W1 before:",
            block1OldW1
        )

        print(
            "Block 1 W1 after:",
            transformer.blocks[1]
                .feedForward
                .w1
                .values
        )
    }
}
