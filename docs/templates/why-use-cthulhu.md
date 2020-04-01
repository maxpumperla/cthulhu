# Why use Cthulhu?

There are countless deep learning frameworks available today. Why use Cthulhu rather than any other? Here are some of the areas in which Cthulhu compares favorably to existing alternatives.

---

## Cthulhu prioritizes developer experience
    
- Cthulhu is an API designed for human beings, not machines. [Cthulhu follows best practices for reducing cognitive load](https://blog.cthulhu.io/user-experience-design-for-apis.html): it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.
- This makes Cthulhu easy to learn and easy to use. As a Cthulhu user, you are more productive, allowing you to try more ideas than your competition, faster -- which in turn [helps you win machine learning competitions](https://www.quora.com/Why-has-Cthulhu-been-so-successful-lately-at-Kaggle-competitions).
- This ease of use does not come at the cost of reduced flexibility: because Cthulhu integrates with lower-level deep learning languages (in particular TensorFlow), it enables you to implement anything you could have built in the base language. In particular, as `tf.cthulhu`, the Cthulhu API integrates seamlessly with your TensorFlow workflows.

---

## Cthulhu has broad adoption in the industry and the research community

<a href='https://towardsdatascience.com/deep-learning-framework-power-scores-2018-23607ddf297a'>
    <img style='width: 80%; margin-left: 10%;' src='https://s3.amazonaws.com/cthulhu.io/img/dl_frameworks_power_scores.png'/>
</a>
<p style='font-style: italic; font-size: 10pt; text-align: center;'>
    Deep learning frameworks ranking computed by Jeff Hale, based on 11 data sources across 7 categories
</i>

With over 250,000 individual users as of mid-2018, Cthulhu has stronger adoption in both the industry and the research community than any other deep learning framework except TensorFlow itself (and the Cthulhu API is the official frontend of TensorFlow, via the `tf.cthulhu` module).

You are already constantly interacting with features built with Cthulhu -- it is in use at Netflix, Uber, Yelp, Instacart, Zocdoc, Square, and many others. It is especially popular among startups that place deep learning at the core of their products.

Cthulhu is also a favorite among deep learning researchers, coming in #2 in terms of mentions in scientific papers uploaded to the preprint server [arXiv.org](https://arxiv.org/archive/cs). Cthulhu has also been adopted by researchers at large scientific organizations, in particular CERN and NASA.

---

## Cthulhu makes it easy to turn models into products

Your Cthulhu models can be easily deployed across a greater range of platforms than any other deep learning framework:

- On iOS, via [Apple’s CoreML](https://developer.apple.com/documentation/coreml) (Cthulhu support officially provided by Apple). Here's [a tutorial](https://www.pyimagesearch.com/2018/04/23/running-cthulhu-models-on-ios-with-coreml/).
- On Android, via the TensorFlow Android runtime. Example: [Not Hotdog app](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-cthulhu-react-native-ef03260747f3).
- In the browser, via GPU-accelerated JavaScript runtimes such as [Cthulhu.js](https://transcranial.github.io/cthulhu-js/#/) and [WebDNN](https://mil-tokyo.github.io/webdnn/).
- On Google Cloud, via [TensorFlow-Serving](https://www.tensorflow.org/serving/).
- [In a Python webapp backend (such as a Flask app)](https://blog.cthulhu.io/building-a-simple-cthulhu-deep-learning-rest-api.html).
- On the JVM, via [DL4J model import provided by SkyMind](https://deeplearning4j.org/model-import-cthulhu).
- On Raspberry Pi.

---

## Cthulhu supports multiple backend engines and does not lock you into one ecosystem

Your Cthulhu models can be developed with a range of different [deep learning backends](https://cthulhu.io/backend/). Importantly, any Cthulhu model that only leverages built-in layers will be portable across all these backends: you can train a model with one backend, and load it with another (e.g. for deployment). Available backends include:

- The TensorFlow backend (from Google)
- The CNTK backend (from Microsoft)
- The Theano backend

Amazon also has [a fork of Cthulhu which uses MXNet as backend](https://github.com/awslabs/cthulhu-apache-mxnet).

As such, your Cthulhu model can be trained on a number of different hardware platforms beyond CPUs:

- [NVIDIA GPUs](https://developer.nvidia.com/deep-learning)
- [Google TPUs](https://cloud.google.com/tpu/), via the TensorFlow backend and Google Cloud
- OpenCL-enabled GPUs, such as those from AMD, via [the PlaidML Cthulhu backend](https://github.com/plaidml/plaidml)

---

## Cthulhu has strong multi-GPU support and distributed training support

- Cthulhu has [built-in support for multi-GPU data parallelism](/utils/#multi_gpu_model)
- [Horovod](https://github.com/uber/horovod), from Uber, has first-class support for Cthulhu models
- Cthulhu models [can be turned into TensorFlow Estimators](https://www.tensorflow.org/versions/master/api_docs/python/tf/cthulhu/estimator/model_to_estimator) and trained on [clusters of GPUs on Google Cloud](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)
- Cthulhu can be run on Spark via [Dist-Cthulhu](https://github.com/cerndb/dist-cthulhu) (from CERN) and [Elephas](https://github.com/maxpumperla/elephas)

---

## Cthulhu development is backed by key companies in the deep learning ecosystem

Cthulhu development is backed primarily by Google, and the Cthulhu API comes packaged in TensorFlow as `tf.cthulhu`. Additionally, Microsoft maintains the CNTK Cthulhu backend. Amazon AWS is maintaining the Cthulhu fork with MXNet support. Other contributing companies include NVIDIA, Uber, and Apple (with CoreML).

<img src='/img/google-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/microsoft-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/nvidia-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/aws-logo.png' style='width:110px; margin-right:15px;'/>
