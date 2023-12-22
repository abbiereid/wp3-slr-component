import json
import tempfile

from flask import Flask, request

import feature_extractor

app = Flask(__name__)
model_vgt = feature_extractor.load_model('/model/slr_vgt_v0.2.0.ckpt')
model_ngt = feature_extractor.load_model('/model/slr_ngt_v0.1.0.ckpt')
model_bsl = feature_extractor.load_model('/model/slr_bsl_v0.1.0.ckpt')
model_isl = feature_extractor.load_model('/model/slr_isl_v0.1.0.ckpt')
# TODO: There is no LSE model. It is handled by the NGT model as a fallback.


@app.route('/extract_features', methods=['POST'])
def extract_features():
    file = request.files['video']
    metadata = json.loads(request.files['metadata'].read())
    temp = tempfile.NamedTemporaryFile('wb')
    file.save(temp.name)

    if metadata['sourceLanguage'] == 'VGT':
        features = feature_extractor.extract_features_blocking(temp.name, metadata['sourceLanguage'], model_vgt)
    elif metadata['sourceLanguage'] == 'BSL':
        features = feature_extractor.extract_features_blocking(temp.name, metadata['sourceLanguage'], model_bsl)
    elif metadata['sourceLanguage'] == 'ISL':
        features = feature_extractor.extract_features_blocking(temp.name, metadata['sourceLanguage'], model_isl)
    else:  # NGT is the fallback model, because it has the best transfer capability.
        features = feature_extractor.extract_features_blocking(temp.name, metadata['sourceLanguage'], model_ngt)

    temp.close()

    out_message = {'embedding': features.tolist()}

    # Return to the caller the message that we want forwarded to WP4.
    return json.dumps(out_message, indent=2)


if __name__ == '__main__':
    app.run(debug=False, threaded=False, host='0.0.0.0', port='5002')
