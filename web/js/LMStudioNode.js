import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.LMStudioNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (["LMStudioGenerate", "LMStudioGenerateAdvance", "LMStudioVision", "LMStudioConnectivityV2", "LMStudioGenerateV2", "LMStudioSequentialPrompt", "LMStudioSequentialPromptAdvanced"].includes(nodeData.name) ) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");

        if (!urlWidget || !modelWidget) {
            console.warn(`LMStudioNode: Could not find 'url' or 'model' widget for node ${nodeData.name}. Skipping model update.`);
            return;
        }

        const fetchModels = async (url) => {
          try {
            const response = await fetch("/lmstudio/get_models", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                url,
              }),
            });

            if (response.ok) {
              const models = await response.json();
              console.debug("Fetched LM Studio models:", models);
              return models;
            } else {
              console.error(`Failed to fetch LM Studio models: ${response.status}`);
              return [];
            }
          } catch (error) {
            console.error(`Error fetching LM Studio models`, error);
            return [];
          }
        };

        const updateModels = async () => {
          const url = urlWidget.value;
          const prevValue = modelWidget.value
          modelWidget.value = ''
          modelWidget.options.values = []

          const models = await fetchModels(url);

          modelWidget.options.values = models;
          console.debug("Updated modelWidget.options.values for LM Studio:", modelWidget.options.values);

          if (models.includes(prevValue)) {
            modelWidget.value = prevValue;
          } else if (models.length > 0) {
            modelWidget.value = models[0];
          }

          console.debug("Updated modelWidget.value for LM Studio:", modelWidget.value);
        };

        urlWidget.callback = updateModels;

        const dummy = async () => {
        }

        await dummy();
        await updateModels();
      };
    }
  },
});
