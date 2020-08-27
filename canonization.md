##Design Ideas
- submodule
    - input: Model
    - Canonizer main class
    - for each subnet:
        - subnet Type checks
            - mapping subnet type --> Replacement Functionality
        - Replacement Functionality (input: subnet of one type; output: replacement subnet)
            - how to insert into model?
            - function/class   
        - insert into return model
        - build return model
    - output: Canonized Model (=return model)
    
    
##Example Code
    ```
    def replace(self, model):
    
            layers = [l for l in model.layers]
            #if no avgPool found, return old model
            flag_layerFound = False
    
            x = layers[0].output
            for i_l, l in enumerate(layers[1:]):
    
                if kchecks.is_average_pooling(l):
    
                    expand_layer, Avg_to_Conv3d, squeeze_layer = self.tranformRule(self, l, i_l)
                    x = expand_layer(x)
                    x = Avg_to_Conv3d(x)
                    x = squeeze_layer(x)
                    flag_layerFound = True
                else:
                    x = l(x)
    
            if flag_layerFound == True:
                # if at least one AvgPool Layer found
                new_model = keras.models.Model(inputs=layers[0].input, outputs=x)
                return new_model
            else:
                # if no avgPool found, return old model
                return model
            
    ```
    ```
    def tranformRule(self, layer, i_l):

        #transforming AvgPool2D
        if isinstance(layer, keras.layers.AveragePooling2D):

            # AvgPooling2D is replaced by a Conv3D Layer
            # Conv2D Layer could also be a good alternative for a AvgPool2D layer
            # BUT the LRP.Flat rule is replacing all weights with 1 and this results in a wrong
            # backward pass in LRP. Conv3D layer is immune to this as the filter size is different.

            Avg_to_Conv3d = keras.layers.Conv3D(1, (2, 2, 1), strides=(2, 2, 1), use_bias=False, activation="linear",
                                                kernel_initializer=keras.initializers.Constant(
                                                    value=1 / (2 * 2)), name=f"AvgPool2D_Conv3D_{i_l}")


            # as input dimension has to be 4 dimensional, we reshape
            expand_layer = keras.layers.Reshape((*layer.input_shape[1:], 1))
            squeeze_layer = keras.layers.Reshape(layer.output_shape[1:])


        # TODO: More layers to support
        else:
            raise Exception("Can only analyze AveragePooling2D. Please contact developers for more functionality")


        return expand_layer, Avg_to_Conv3d, squeeze_layeronal, we reshape
            expand_layer = keras.layers.Reshape(
    ```