# Source: https://rpubs.com/Joaquin_AR/406480

library(h2o)

# Creación de un cluster local con todos los cores disponibles.
h2o.init(ip = "localhost",
         # -1 indica que se empleen todos los cores disponibles.
         nthreads = -1,
         # Máxima memoria disponible para el cluster.
         max_mem_size = "4g")

# Se eliminan los datos del cluster por si ya había sido iniciado.
h2o.removeAll()

# Carga de datos en R y transferencia a H2O.
url_path <- paste0("https://github.com/JoaquinAmatRodrigo/Estadistica-con-R/raw/",
                   "master/datos/adult_custom.csv")
datos_R   <- read.csv(file = url_path, header = TRUE)
datos_h2o <- as.h2o(x = datos_R, destination_frame = "datos_h2o")

# Dimensiones del set de datos
h2o.dim(datos_h2o)

# Nombre de las columnas
h2o.colnames(datos_h2o)

h2o.describe(datos_h2o)

# Índices
indices <- h2o.columns_by_type(object = datos_h2o, coltype = "numeric")
indices

# Nombres
h2o.colnames(x = datos_h2o)[indices]

indices <- h2o.columns_by_type(object = datos_h2o, coltype = "numeric")
h2o.cor(x = datos_h2o[,indices], y = NULL, na.rm = TRUE)

# Se crea una tabla con el número de observaciones de cada tipo.
tabla_muestra <- as.data.frame(h2o.table(datos_h2o$salario))
tabla_muestra

# Una vez creada la tabla, se carga en el entorno de R para poder graficar.
library(tidyverse)
ggplot(data = tabla_muestra,
       aes(x = salario, y = Count, fill = salario)) +
  geom_col() +
  scale_fill_manual(values = c("gray50", "orangered2")) +
  theme_bw() +
  labs(x = "Salario", y = "Número de observaciones",
       title = "Distribución de la variable Salario") +
  theme(legend.position = "none")

# Separación de las observaciones en conjunto de entrenamiento y test.
# En los ejemplos de GBM y deep learning se repetirá la separación, pero en 
# tres conjuntos en lugar de dos.
particiones <- h2o.splitFrame(data = datos_h2o, ratios = c(0.8), seed = 123)
datos_train_h2o   <- h2o.assign(data = particiones[[1]], key = "datos_train_H2O")
datos_test_h2o    <- h2o.assign(data = particiones[[2]], key = "datos_test_H2O")

h2o.table(datos_train_h2o$salario)

h2o.table(datos_test_h2o$salario)

# En porcentaje
h2o.table(datos_train_h2o$salario)/h2o.nrow(datos_train_h2o)

# Se comprueba que la variable respuesta es de tipo factor.
datos_train_h2o$salario <- h2o.asfactor(datos_train_h2o$salario)
datos_test_h2o$salario <- h2o.asfactor(datos_test_h2o$salario)
h2o.isfactor(datos_train_h2o$salario)

# Se define la variable respuesta y los predictores.
var_respuesta <- "salario"
# Para este modelo se emplean todos los predictores disponibles.
predictores   <- setdiff(h2o.colnames(datos_h2o), var_respuesta)

# Ajuste del modelo y validación mediente 5-CV para estimar su error.
modelo_binomial <- h2o.glm(
  y = var_respuesta,
  x = predictores,
  training_frame = datos_train_h2o,
  family = "binomial",
  link = "logit",
  standardize = TRUE,
  balance_classes = FALSE,
  ignore_const_cols = TRUE,
  # Se especifica que hacer con observaciones incompletas
  missing_values_handling = "Skip",
  # Se hace una búsqueda del hiperparámetro lamba.
  lambda_search = TRUE,
  # Selección automática del solver adecuado.
  solver = "AUTO",
  alpha = 0.95,
  # Validación cruzada de 5 folds para estimar el error
  # del modelo.
  seed = 123,
  nfolds = 5,
  # Reparto estratificado de las observaciones en la creación
  # de las particiones.
  fold_assignment = "Stratified",
  keep_cross_validation_predictions = FALSE,
  model_id = "modelo_binomial"
)

modelo_binomial

# Coeficientes de regresión de cada uno de los predictores.
modelo_binomial@model$coefficients_table

# Para mostrarlos todos.
# as.data.frame(modelo_binomial@model$coefficients_table)

# Predictores incluidos.
names(modelo_binomial@model$coefficients[modelo_binomial@model$coefficients > 0])

coeficientes <- as.data.frame(modelo_binomial@model$coefficients_table)

# Se excluye el intercept.
coeficientes <- coeficientes %>% filter(names != "Intercept")

# Se calcula el valor absoluto.
coeficientes <- coeficientes %>%
  mutate(abs_stand_coef = abs(standardized_coefficients))

# Se añade una variable con el signo del coeficiente.
coeficientes <- coeficientes %>%
  mutate(signo = if_else(standardized_coefficients > 0,
                         "Positivo",
                         "Negativo"))

ggplot(data = coeficientes,
       aes(x = reorder(names, abs_stand_coef),
           y = abs_stand_coef,
           fill = signo)) +
  geom_col() +
  coord_flip() +
  labs(title = "Importancia de los predictores en el modelo GLM",
       x = "Predictor",
       y = "Valor absoluto coeficiente estandarizado") +
  theme_bw() +
  theme(legend.position = "bottom")

# Equivalente:
# h2o.varimp(modelo_binomial)
# h2o.varimp_plot(modelo_binomial)

# Área bajo la curva
h2o.auc(modelo_binomial, train = TRUE)

# Mean Squared Error
h2o.mse(modelo_binomial, train = TRUE)
# R2
h2o.r2(modelo_binomial, train = TRUE)
# LogLoss
h2o.logloss(modelo_binomial, train = TRUE)
# Coeficiente de Gini
h2o.giniCoef(modelo_binomial, train = TRUE)
# Desviance del modelo nulo
h2o.null_deviance(modelo_binomial, train = TRUE)
# Desviance del modelo final
h2o.residual_deviance(modelo_binomial, train = TRUE)
# AIC
h2o.aic(modelo_binomial, train = TRUE)

h2o.performance(model = modelo_binomial, train = TRUE)

# Equivalente a:
# modelo_binomial@model$training_metrics

h2o.performance(model = modelo_binomial, train = TRUE)

# Área bajo la curva
h2o.auc(modelo_binomial, xval = TRUE)

h2o.performance(model = modelo_binomial, xval = TRUE)

predicciones <- h2o.predict(object = modelo_binomial, newdata = datos_test_h2o)
predicciones

h2o.performance(model = modelo_binomial, newdata = datos_test_h2o)

# Cálculo manual de accuracy
mean(as.vector(predicciones$predict) == as.vector(datos_test_h2o$salario))

# Valores de alpha que se van a comparar.
hiperparametros <- list(alpha = c(0, 0.1, 0.5, 0.95, 1))

grid_glm <- h2o.grid(
  # Algoritmo y parámetros.
  algorithm = "glm",
  family = "binomial",
  link = "logit",
  # Variable respuesta y predictores.
  y = var_respuesta,
  x = predictores,
  # Datos de entrenamiento.
  training_frame = datos_train_h2o,
  # Preprocesado.
  standardize = TRUE,
  missing_values_handling = "Skip",
  ignore_const_cols = TRUE,
  # Hiperparámetros.
  hyper_params = hiperparametros,
  # Tipo de búsqueda.
  search_criteria = list(strategy = "Cartesian"),
  lambda_search = TRUE,
  # Selección automática del solver adecuado.
  solver = "AUTO",
  # Estrategia de validación para seleccionar el mejor modelo.
  seed = 123,
  nfolds = 10,
  # Reparto estratificado de las observaciones en la creación
  # de las particiones.
  fold_assignment = "Stratified",
  keep_cross_validation_predictions = FALSE,
  grid_id = "grid_glm"
)

# Se muestran los modelos ordenados de mayor a menor AUC.
resultados_grid <- h2o.getGrid(grid_id = "grid_glm",
                               sort_by = "auc",
                               decreasing = TRUE)
print(resultados_grid)

# Identificador de los modelos creados por validación cruzada.
id_modelos <- unlist(resultados_grid@model_ids)

# Se crea una lista donde se almacenarán los resultados.
auc_xvalidacion <- vector(mode = "list", length = length(id_modelos))

# Se recorre cada modelo almacenado en el grid y se extraen la métrica (auc)
# obtenida en cada partición.
for (i in seq_along(id_modelos)) {
  modelo <- h2o.getModel(resultados_grid@model_ids[[i]])
  metricas_xvalidacion_modelo <- modelo@model$cross_validation_metrics_summary
  names(auc_xvalidacion)[i]   <- modelo@model$model_summary$regularization
  auc_xvalidacion[[i]] <- as.numeric(metricas_xvalidacion_modelo["auc", -c(1,2)])
}

# Se eliminan los espacios en blanco del nombre de los modelos.
library(stringr)
names(auc_xvalidacion) <- str_remove_all(string = names(auc_xvalidacion),
                                         pattern = "[ )=]")
names(auc_xvalidacion) <- str_replace_all(string = names(auc_xvalidacion),
                                          pattern = "[(,]",
                                          replacement = "_")
# Se convierte la lista en dataframe.
auc_xvalidacion_df <- as.data.frame(auc_xvalidacion) %>%
  mutate(resample = row_number()) %>% 
  gather(key = "modelo", value = "auc", -resample) %>%
  mutate(modelo = str_replace_all(string = modelo,
                                  pattern = "_" ,
                                  replacement = " \n "))
# Gráfico
ggplot(data = auc_xvalidacion_df, aes(x = modelo, y = auc, color = modelo)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.6) +
  stat_summary(fun.y = "mean", colour = "red", size = 2, geom = "point") +
  theme_bw() +
  labs(title = "Accuracy obtenido por 10-CV") +
  coord_flip() +
  theme(legend.position = "none")

# Se extrae el mejor modelo, que es el que ocupa la primera posición tras haber
# ordenado los resultados de mayor a menor AUC (métrica utilizada en este caso).
modelo_glm_final <- h2o.getModel(resultados_grid@model_ids[[1]])

particiones <- h2o.splitFrame(data = datos_h2o, ratios = c(0.6, 0.20),
                              seed = 123)
datos_train_h20 <- h2o.assign(data = particiones[[1]], key = "datos_train_H2O")
datos_val_h20   <- h2o.assign(data = particiones[[2]], key = "datos_val_H2O")
datos_test_h20  <- h2o.assign(data = particiones[[3]], key = "datos_test_H2O")

modelo_gbm <- h2o.gbm(
  # Tipo de distribución (clasificación binaria)
  distribution = "bernoulli",
  # Variable respuesta y predictores.
  y = var_respuesta,
  x = predictores,
  # Datos de entrenamiento.
  training_frame = datos_train_h20,
  # Datos de validación para estimar el error.
  validation_frame = datos_val_h20,
  # Número de árboles.
  ntrees = 500,
  # Complejidad de los árboles
  max_depth = 3,
  min_rows = 10,
  # Aprendizaje
  learn_rate = 0.01,
  # Detención temprana
  score_tree_interval = 5,
  stopping_rounds = 3,
  stopping_metric = "AUC",
  stopping_tolerance = 0.001,
  model_id = "modelo_gbm",
  seed = 123)

modelo_gbm

# H2O almacena las métricas de entrenamiento y test bajo el nombre de scoring.
# Los valores se encuentran almacenados dentro del modelo.
scoring <- as.data.frame(modelo_gbm@model$scoring_history)
head(scoring)

scoring <- scoring %>%
  select(-timestamp, -duration) %>%
  gather(key = "metrica", value = "valor", -number_of_trees) %>%
  separate(col = metrica, into = c("conjunto", "metrica"), sep = "_")

scoring %>% filter(metrica == "auc") %>%
  ggplot(aes(x = number_of_trees, y = valor, color = conjunto)) +
  geom_line() +
  labs(x = "AUC", y = "número de árboles (ntrees)",
       title = "Evolución del AUC vs número de árboles") +
  theme_bw()

# Se extraen los valores de importancia
importancia <- as.data.frame(modelo_gbm@model$variable_importances)
importancia

ggplot(data = importancia,
       aes(x = reorder(variable, scaled_importance),
           y = scaled_importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Importancia de los predictores en el modelo GBM",
       subtitle = "Importancia en base a la reducción del error cuadrático medio",
       x = "Predictor",
       y = "Importancia relativa") +
  theme_bw()

predicciones <- h2o.predict(object = modelo_gbm, newdata = datos_test_h2o)
predicciones

# AUC de test
h2o.performance(model = modelo_gbm, newdata = datos_test_h2o)@metrics$AUC

# Hiperparámetros que se quieren comparar.
hiperparametros <- list(learn_rate  = c(0.001, 0.05, 0.1),
                        max_depth   = c(5, 10, 15, 20),
                        sample_rate = c(0.8, 1))

grid_gbm <- h2o.grid(
  # Algoritmo.
  algorithm = "gbm",
  distribution = "bernoulli",
  # Variable respuesta y predictores.
  y = var_respuesta,
  x = predictores,
  # Datos de entrenamiento.
  training_frame = datos_train_h2o,
  # Datos de validación.
  validation_frame = datos_val_h20,
  # Preprocesado.
  ignore_const_cols = TRUE,
  # Detención temprana.
  score_tree_interval = 10,
  stopping_rounds = 3,
  stopping_metric = "AUC",
  stopping_tolerance = 0.001,
  # Hiperparámetros fijados.
  ntrees = 5000,
  min_rows = 10,
  # Hiperparámetros optimizados.
  hyper_params = hiperparametros,
  # Tipo de búsqueda.
  search_criteria = list(strategy = "Cartesian"),
  seed = 123,
  grid_id = "grid_gbm"
)

# Se muestran los modelos ordenados de mayor a menor AUC.
resultados_grid <- h2o.getGrid(grid_id = "grid_gbm",
                               sort_by = "auc",
                               decreasing = TRUE)

data.frame(resultados_grid@summary_table) %>% select(-model_ids)

modelo_gbm_final <- h2o.getModel(resultados_grid@model_ids[[1]])

# AUC de test
h2o.performance(model = modelo_gbm_final, newdata = datos_test_h2o)@metrics$AUC

# Hiperparámetros que se quieren optimizar mediante búsqueda aleatoria.
# Para realizar esta búsqueda se tiene que pasar un vector de posibles valores
# de cada hiperparámetro, entre los que se escoge aleatoriamente.
hiperparametros <- list( 
  min_rows = seq(from = 5, to = 50, by = 10),                           
  nbins = 2^seq(4, 10, 1),                                                     
  nbins_cats = 2^seq(4, 12, 1),
  min_split_improvement = c(0, 1e-8, 1e-6, 1e-4),
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin") 
)

# Al ser una búsqueda aleatoria, hay que indicar criterios de parada.
search_criteria <- list(
  strategy = "RandomDiscrete",      
  # Tiempo máximo de búsqueda (5 minutos).
  max_runtime_secs = 5*60,         
  # Número máximo de modelos.
  max_models = 100,                  
  # Reproducible.
  seed = 1234               
)

grid_gbm2 <- h2o.grid(
  # Algoritmo.
  algorithm = "gbm",
  distribution = "bernoulli",
  # Variable respuesta y predictores.
  y = var_respuesta,
  x = predictores,
  # Datos de entrenamiento.
  training_frame = datos_train_h2o,
  # Datos de validación.
  validation_frame = datos_val_h20,
  # Preprocesado.
  ignore_const_cols = TRUE,
  # Detención temprana
  score_tree_interval = 10,
  stopping_rounds = 3,
  stopping_metric = "AUC",
  stopping_tolerance = 0.001,
  # Hiperparámetros fijados
  ntrees = 5000,
  learn_rate = 0.1, 
  max_depth = 20,
  sample_rate = 0.8, 
  # Hiperparámetros optimizados.
  hyper_params = hiperparametros,
  # Tipo de búsqueda.
  search_criteria = search_criteria,
  seed = 123,
  grid_id = "grid_gbm2"
)

# Se muestran los modelos ordenados de mayor a menor AUC.
resultados_grid <- h2o.getGrid(grid_id = "grid_gbm2",
                               sort_by = "auc",
                               decreasing = TRUE)

data.frame(resultados_grid@summary_table) %>% select(-model_ids)

modelo_gbm_final <- h2o.getModel(resultados_grid@model_ids[[1]])

# AUC de test
h2o.performance(model = modelo_gbm_final, newdata = datos_test_h2o)@metrics$AUC

# Número de observaciones por clase
N = 300
# Número de dimensiones
D = 2
# Número clases
K = 3
# Matriz para almacenar las observaciones
x_1 = vector(mode = "numeric")
x_2 = vector(mode = "numeric")
y   = vector(mode = "numeric")

# Simulación de los datos
for(i in 1:K){
  set.seed(123)
  r = seq(from = 0, to = 1, length.out = N) 
  t = seq(from =  i*4, to = (i+1)*4, length.out = N) + rnorm(n = N) * 0.35
  x_1 <- c(x_1, r * sin(t))
  x_2 <- c(x_2, r*cos(t))
  y   <- c(y, rep(letters[i], N))
}

datos_espiral <- data.frame(y, x_1, x_2)
ggplot(data = datos_espiral, aes(x = x_1, y = x_2, color = y)) + 
  geom_point() +
  theme_bw() + 
  theme(legend.position = "none",
        axis.text = element_blank())

datos_espiral_h2o <- as.h2o(datos_espiral)

modelo_dl_10 <- h2o.deeplearning(x = c("x_1", "x_2"),
                                 y = "y",
                                 distribution = "multinomial",
                                 training_frame = datos_espiral_h2o,
                                 standardize = TRUE,
                                 activation = "Rectifier",
                                 hidden = 10,
                                 stopping_rounds = 0,
                                 epochs = 50,
                                 seed = 123,
                                 model_id = "modelo_dl_10"
)

modelo_dl_10_1k <- h2o.deeplearning(x = c("x_1", "x_2"),
                                    y = "y",
                                    distribution = "multinomial",
                                    training_frame = datos_espiral_h2o,
                                    standardize = TRUE,
                                    activation = "Rectifier",
                                    hidden = 10,
                                    stopping_rounds = 0,
                                    epochs = 1000,
                                    seed = 123,
                                    model_id = "modelo_dl_10_5k"
)

modelo_dl_2_200 <- h2o.deeplearning(x = c("x_1", "x_2"),
                                    y = "y",
                                    distribution = "multinomial",
                                    training_frame = datos_espiral_h2o,
                                    standardize = TRUE,
                                    activation = "Rectifier",
                                    hidden = c(200, 200),
                                    stopping_rounds = 0,
                                    epochs = 1000,
                                    seed = 123,
                                    model_id = "modelo_dl_100"
)

grid_predicciones <- expand.grid(x_1 = seq(from = -1, to = 1, length = 75),
                                 x_2 = seq(from = -1, to = 1, length = 75))
grid_predicciones_h2o <- as.h2o(grid_predicciones)

predicciones_10 <- h2o.predict(object = modelo_dl_10,
                               newdata = grid_predicciones_h2o)
grid_predicciones$y_10 <- as.vector(predicciones_10$predict)

predicciones_10_1k <- h2o.predict(object = modelo_dl_10_1k,
                                  newdata = grid_predicciones_h2o)
grid_predicciones$y_10_1k <- as.vector(predicciones_10_1k$predict)

predicciones_2_200 <- h2o.predict(object = modelo_dl_2_200,
                                  newdata = grid_predicciones_h2o)
grid_predicciones$y_2_200 <- as.vector(predicciones_2_200$predict)

ggplot(data = grid_predicciones, aes(x = x_1, y = x_2, color = y_10)) + 
  geom_point(size = 0.5) +
  theme_bw() + 
  labs(title = "1 capa oculta, 10 neuronas, 50 epochs") +
  theme(legend.position = "none",
        axis.text = element_blank())

ggplot(data = grid_predicciones, aes(x = x_1, y = x_2, color = y_10_1k)) + 
  geom_point(size = 0.5) +
  labs(title = "1 capa oculta, 10 neuronas, 1000 epochs") +
  theme_bw() + 
  theme(legend.position = "none",
        axis.text = element_blank())

ggplot(data = grid_predicciones, aes(x = x_1, y = x_2, color = y_2_200)) + 
  geom_point(size = 0.5) +
  labs(title = "2 capa oculta, 200 neuronas, 1000 epochs") +
  theme_bw() + 
  theme(legend.position = "none",
        axis.text = element_blank())

# Hiperparámetros que se quieren comparar.
hiperparametros <- list(hidden = list(c(64), c(128), c(256), c(512), c(1024),
                                      c(64,64), c(128,128), c(256,256),
                                      c(512, 512)))
grid_dl <- h2o.grid(
  # Algoritmo.
  algorithm = "deeplearning",
  activation = "RectifierWithDropout",
  epochs = 500,
  # Variable respuesta y predictores.
  y = var_respuesta,
  x = predictores,
  # Datos de entrenamiento.
  training_frame = datos_train_h2o,
  shuffle_training_data = FALSE,
  # Datos de validación.
  validation_frame = datos_val_h20,
  # Preprocesado.
  standardize = TRUE,
  missing_values_handling = "Skip",
  # Detención temprana.
  stopping_rounds = 3,
  stopping_metric = "AUC",
  stopping_tolerance = 0.01,
  # Hiperparámetros optimizados.
  hyper_params = hiperparametros,
  # Regularización
  l1 = 1e-5,
  l2 = 1e-5,
  # Tipo de búsqueda.
  search_criteria = list(strategy = "Cartesian"),
  seed = 123,
  grid_id = "grid_dl"
)

# Se muestran los modelos ordenados de mayor a menor AUC.
resultados_grid <- h2o.getGrid(grid_id = "grid_dl",
                               sort_by = "auc",
                               decreasing = TRUE)

data.frame(resultados_grid@summary_table) %>% select(-model_ids)

# Se muestran los modelos ordenados de mayor a menor AUC.
resultados_grid <- h2o.getGrid(grid_id = "grid_dl",
                               sort_by = "auc",
                               decreasing = TRUE)

data.frame(resultados_grid@summary_table) %>% select(-model_ids)

modelo_dl_final <- h2o.getModel(resultados_grid@model_ids[[1]])
plot(modelo_dl_final, timestep = "epochs", metric = "classification_error")

# AUC de test
h2o.performance(model = modelo_dl_final, newdata = datos_test_h2o)@metrics$AUC

h2o.partialPlot(object = modelo_gbm_final,
                data = datos_h2o,
                cols = c("age"),
                plot = TRUE,
                plot_stddev = TRUE)

predict(modelo_gbm_final, newdata = datos_h2o[1,])

# Función predict especial para un modelo H2O de clasificación binaria.
predict_custom <- function(object, newdata){
  h2o.no_progress()
  as.vector(predict(object, newdata = as.h2o(newdata))[[3]])
}

# Ejemplo
predict_custom(object = modelo_gbm_final, newdata = datos_h2o[1,])

library(pdp)

calcular_graficar_ice <- function(predictor, modelo_h2o, funcion_predict,
                                  grid.resolution, datos_train_h2o, fraccion){
  # Función para calcular y graficar curvas ICE de uno o varios predictores
  
  datos_train <- as.data.frame(datos_h2o)
  datos_train <- dplyr::sample_frac(tbl = datos_train, size = fraccion)
  plot_ice <- partial(
    pred.var = predictor,
    object = modelo_h2o,
    pred.fun = funcion_predict,
    grid.resolution = grid.resolution, 
    train = datos_train,
    ice = TRUE,
    center = TRUE,
    quantiles = FALSE,
    plot = TRUE,
    rug = TRUE,
    alpha = 0.1,
    parallel = TRUE,
    plot.engine = "ggplot2"
  )
  plot_ice <- plot_ice + geom_rug(data = datos_train,
                                  aes(x = !!sym(predictor)),
                                  inherit.aes = FALSE)
  return(plot_ice)
}

graficos_ice <- map(.x = c("age", "education_number"),
                    .f = calcular_graficar_ice,
                    modelo_h2o = modelo_gbm_final,
                    funcion_predict = predict_custom,
                    grid.resolution = 30, 
                    datos_train_h2o = datos_h2o,
                    fraccion = 0.7)
graficos_ice

# Se guarda el modelo en el directorio actual
h2o.saveModel(object = modelo_dl_final, path = getwd(), force = TRUE)

modelo <- h2o.loadModel(path = "./grid_dl_model_8")

# Se apaga el cluster H2O
h2o.shutdown(prompt = FALSE)