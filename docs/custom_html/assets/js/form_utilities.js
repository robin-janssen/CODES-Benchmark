
var customModelIDs = [];
const surrogateNames = {
    "model-fcnn": "FullyConnected",
    "model-deeponet": "MultiONet",
    "model-latent_node": "LatentNeuralODE",
    "model-latent_poly": "LatentPolynomial"
}

function toggleElementVisible(textFieldId, isChecked) {
    var textField = document.getElementById(textFieldId);
    textField.style.display = isChecked ? "block" : "none";
}

function getNextFreeModelID() {
    for (var i = 1; i <= 10; i++) {
        if (customModelIDs.includes(i) == false) {
            return i;
        }
    }
    return -1;
}

function addCustomModel() {
    if (customModelIDs.length < 10) {
        const customModelID = getNextFreeModelID();
        var container = document.getElementById("custom-models-container");
        var textField = document.createElement("div");
        textField.innerHTML = `
					<ul class="actions fit">
						<li><input type="text" name="custom-model-${customModelID}" id="custom-model-${customModelID}" value="" placeholder="Custom Model ${customModelID}" /></li>
						<li style="width: 20%; display: flex; align-items: center;"><button type="button" onclick="removeCustomModel(this)">Remove</button></li>
					</ul>
					<ul class="actions fit">
						<li><input type="text" name="custom-model-${customModelID}-batchsize" id="custom-model-${customModelID}-batchsize" value="" placeholder="Batch size"></li>
						<li><input type="text" name="custom-model-${customModelID}-epochs" id="custom-model-${customModelID}-epochs" value="" placeholder="Number of epochs"></li>
					</ul>
					`;
        container.appendChild(textField);
        customModelIDs.push(customModelID);
    }
}

function removeCustomModel(button) {
    var id = button.parentElement.parentElement.children[0].children[0].id;
    var index = customModelIDs.indexOf(parseInt(id.substring(13)));
    customModelIDs.splice(index, 1);
    button.parentElement.parentElement.parentElement.remove();
}

function removeAllCusomModels() {
    var container = document.getElementById("custom-models-container");
    container.innerHTML = "";
    customModelIDs = [];
}

function fullResetForm() {

    removeAllCusomModels();

    var customModelsContainer = document.getElementById("custom-models-container");
    customModelsContainer.innerHTML = "";

    var checkboxes = document.querySelectorAll("input[type='checkbox']");
    for (var i = 0; i < checkboxes.length; i++) {
        checkboxes[i].checked = false;
        var id = checkboxes[i].id;
        var divId = "div-" + id.substring(6) + "_params";
        var div = document.getElementById(divId);
        if (div) {
            div.style.display = "none";
        }
    }
    var form = document.getElementById("config-form");
    form.reset();
}

function validateConfigForm() {
    const requiredFields = ["training_id", "data-dataset", "misc-seed"];
    const requiredFieldLabels = ["Training ID", "Dataset", "Seed"];
    // const surrogateLabels = ["Fully connected NN", "DeepONet", "Latent Neural ODE", "Latent Polynomial"];
    for (var i = 0; i < requiredFields.length; i++) {
        var field = document.getElementById(requiredFields[i]);
        if (field.value == "") {
            alert("Field " + requiredFieldLabels[i] + " is required");
            return false;
        }
    }

    const [surrogates, surrogateParams] = getSelectedSurrogatesWithParams();
    const [customModels, customModelParams] = getCustomModelsWithParams();
    if (surrogates.length == 0 && customModels.length == 0) {
        alert("At least one surrogate model must be selected");
        return false;
    }
    for (var i = 0; i < surrogates.length; i++) {
        const batchsize = surrogateParams[surrogates[i]].batchsize;
        const epochs = surrogateParams[surrogates[i]].epochs;
        if (batchsize == "" || isNaN(batchsize)) {
            alert("Batch size for " + surrogateNames[surrogates[i]] + " is required and must be a number");
            return false;
        }
        if (epochs == "" || isNaN(epochs)) {
            alert("Number of epochs for " + surrogateNames[surrogates[i]] + " is required and must be a number");
            return false;
        }
    }

    for (var i = 0; i < customModels.length; i++) {
        const model = customModels[i];
        if (model == "") {
            continue;
        }
        const batchsize = customModelParams[customModels[i]].batchsize;
        const epochs = customModelParams[customModels[i]].epochs;
        if (batchsize == "" || isNaN(batchsize)) {
            alert("Batch size for " + customModels[i] + " is required and must be a number");
            return false;
        }
        if (epochs == "" || isNaN(epochs)) {
            alert("Number of epochs for " + customModels[i] + " is required and must be a number");
            return false;
        }
    }

    if (document.getElementById("bench-interpolation").checked) {
        const intervals = document.getElementById("bench-interpol_ts").value;
        if (intervals == "") {
            alert("Interpolation intervals are required when active");
            return false;
        }
        if (intervals.split(",").some(interval => isNaN(interval))) {
            alert("Interpolation intervals must be numbers");
            return false;
        }
    }

    if (document.getElementById("bench-extrapolation").checked) {
        const cutoffs = document.getElementById("bench-extrapol_cutoffs").value;
        if (cutoffs == "") {
            alert("Extrapolation cutoffs are required when active");
            return false;
        }
        if (cutoffs.split(",").some(cutoff => isNaN(cutoff))) {
            alert("Extrapolation cutoffs must be numbers");
            return false;
        }
    }

    if (document.getElementById("bench-sparse").checked) {
        const factors = document.getElementById("bench-sparse_factors").value;
        if (factors == "") {
            alert("Sparse factors are required when active");
            return false;
        }
        if (factors.split(",").some(factor => isNaN(factor))) {
            alert("Sparse factors must be numbers");
            return false;
        }
    }

    if (document.getElementById("bench-batch_scaling").checked) {
        const batch_scaling = document.getElementById("bench-batch_scaling_factors").value;
        if (batch_scaling == "") {
            alert("Batch scaling factors are required when active");
            return false;
        }
        if (batch_scaling.split(",").some(factor => isNaN(factor))) {
            alert("Batch scaling factors must be numbers");
            return false;
        }
    }

    if (document.getElementById("bench-uq").checked) {
        const uq_samples = document.getElementById("bench-uq_ensemble_size").value;
        if (uq_samples == "") {
            alert("Uncertainty quantification number of samples are required when active");
            return false;
        }
        if (isNaN(uq_samples)) {
            alert("Uncertainty quantification samples must be a number");
            return false;
        }
    }

    const seed = document.getElementById("misc-seed").value;
    if (seed == "") {
        alert("Seed is required");
        return false;
    }
    if (isNaN(seed)) {
        alert("Seed must be a number");
        return false;
    }
    return true;
}


function downloadYAML() {

    if (!validateConfigForm()) {
        return;
    }
    const trainingId = `${document.getElementById("training_id").value}`;
    const datasetName = `${document.getElementById("data-dataset").value}`;

    const config = {
        training_id: String(trainingId),
        surrogates: getSurrogateListString(),
        use_optimal_params: document.getElementById("misc-use_optimal_params").checked,
        dataset: {
            name: String(datasetName),
            log10_transform: document.getElementById("data-log10").checked,
            normalise: String(document.getElementById("data-norm").value)
        },
        devices: getSelectedDevices(),
        seed: Number(document.getElementById("misc-seed").value),
        verbose: document.getElementById("misc-verbose").checked,
        batch_size: getBatchSizeList(),
        epochs: getEpochList(),

        interpolation: {
            enabled: document.getElementById("bench-interpolation").checked,
            intervals: getInterpolationIntervals()
        },
        extrapolation: {
            enabled: document.getElementById("bench-extrapolation").checked,
            cutoffs: getExtrapolationCutoffs()
        },
        sparse: {
            enabled: document.getElementById("bench-sparse").checked,
            factors: getSparseFactors()
        },
        batch_scaling: {
            enabled: document.getElementById("bench-batch_scaling").checked,
            sizes: getBatchScalingFactors()
        },
        uncertainty: {
            enabled: document.getElementById("bench-uq").checked,
            ensemble_size: Number(document.getElementById("bench-uq_ensemble_size").value),
        },
        
        losses: document.getElementById("bench-losses").checked,
        gradients: document.getElementById("bench-dyn_acc").checked,
        timing: document.getElementById("bench-timing").checked,
        compute: document.getElementById("bench-compute").checked,
        compare: document.getElementById("misc-compare").checked,
        
    };

    function Format(data) {
        if (!(this instanceof Format)) return new Format(data);
        this.data = data;
    }

    let CustomFormatType = new jsyaml.Type('!format', {
        kind: 'scalar',
        resolve: () => false,
        instanceOf: Format,
        represent: f => f.data
    });

    let schema = jsyaml.DEFAULT_SCHEMA.extend({ implicit: [CustomFormatType] });

    function replacer(key, value) {
        if (Array.isArray(value) && !value.filter(x => typeof x !== 'number').length) {
            return Format(jsyaml.dump(value, { flowLevel: 0 }).trim());
        }
        return value;
    }

    const yamlConfig = jsyaml.dump(config,
        {
            schema: schema,
            replacer: replacer,
            forceQuotes: true,
            quotingType: '"',
            flowLevel: 2,
        });

    const element = document.createElement("a");
    element.setAttribute("href", "data:text/yaml;charset=utf-8," + encodeURIComponent(yamlConfig));
    element.setAttribute("download", "config.yaml");
    element.style.display = "none";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

function getSelectedSurrogatesWithParams() {
    const surrogates = getSelectedSurrogates();
    const surrogateParams = {};
    surrogates.forEach(surrogate => {
        const batchsize = document.getElementById(`${surrogate}_batchsize`).value;
        const epochs = document.getElementById(`${surrogate}_epochs`).value;
        surrogateParams[surrogate] = {
            batchsize: batchsize,
            epochs: epochs
        };
    });
    return [surrogates, surrogateParams];
}

function getSelectedSurrogates() {
    const surrogates = [];
    const surrogateCheckboxes = document.querySelectorAll('input[id^="model-"]');
    surrogateCheckboxes.forEach(checkbox => {
        if (checkbox.checked) {
            surrogates.push(checkbox.id);
        }
    });
    return surrogates;
}

function getCustomModelsWithParams() {
    const customModels = [];
    const customModelParams = {};
    customModelIDs.forEach(id => {
        const model = document.getElementById(`custom-model-${id}`).value;
        const batchsize = document.getElementById(`custom-model-${id}-batchsize`).value;
        const epochs = document.getElementById(`custom-model-${id}-epochs`).value;
        customModels.push(model);
        customModelParams[model] = {
            batchsize: batchsize,
            epochs: epochs
        };
    });
    return [customModels, customModelParams];
}

function getSelectedDevices() {
    const deviceString = document.getElementById("misc-devices").value;
    if (!deviceString) {
        return "cpu";
    }
    return deviceString.split(",").map(device => device.trim());
}

function getInterpolationIntervals() {
    return getIntListFromElement("bench-interpol_ts");
}

function getExtrapolationCutoffs() {
    return getIntListFromElement("bench-extrapol_cutoffs");
}

function getSparseFactors() {
    return getIntListFromElement("bench-sparse_factors");
}
function getBatchScalingFactors() {
    return getIntListFromElement("bench-batch_scaling_factors");
}

function getIntListFromElement(elementId) {
    const element = document.getElementById(elementId);
    const value = element.value;
    if (value) {
        return value.split(",").map(val => parseInt(val.trim()));
    }
    return [];
}

function getSurrogateListString() {
    const customSurrogates = getCustomModelsWithParams()[0];
    customSurrogates.forEach(model => {
        surrogateNames[model] = model;
    });
    const surrogates = getSelectedSurrogates();
    surrogates.push(...getCustomModelsWithParams()[0]);
    return surrogates.map(surrogate => surrogateNames[surrogate]);
}

function getBatchSizeList() {
    const [surrogates, surrogateParams] = getSelectedSurrogatesWithParams();
    const [customModels, customModelParams] = getCustomModelsWithParams();
    const batchSizes = [];
    surrogates.forEach(surrogate => {
        batchSizes.push(Number(surrogateParams[surrogate].batchsize));
    });
    customModels.forEach(model => {
        batchSizes.push(Number(customModelParams[model].batchsize));
    });
    return batchSizes;
}

function getEpochList() {
    const [surrogates, surrogateParams] = getSelectedSurrogatesWithParams();
    const [customModels, customModelParams] = getCustomModelsWithParams();
    const epochs = [];
    surrogates.forEach(surrogate => {
        epochs.push(Number(surrogateParams[surrogate].epochs));
    });
    customModels.forEach(model => {
        epochs.push(Number(customModelParams[model].epochs));
    });
    return epochs;
}