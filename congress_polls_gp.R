# Load various libraries
library(rjson)
library(mvtnorm)

# Scrape data
congress_page = readLines('http://www.realclearpolitics.com/epolls/json/903_historical.js?1453388629140&callback=return_json')
congress_page_formatted = substring(congress_page, 13,nchar(congress_page)-2)
congress_page_json = fromJSON(congress_page_formatted)
congress_polls = congress_page_json

# Clean data
congress_getDate = function(rowNumber) {
  date.string = substring(congress_polls$poll$rcp_avg[rowNumber][[1]]$date,6,16)
  return(as.Date(date.string, "%d %B %Y"))
}
congress_getApproval = function(rowNumber) {
  return(congress_polls$poll$rcp_avg[rowNumber][[1]]$candidate[[1]]$value)
}

# Use last 1000 days
ndays = 1000

congress_df = data.frame(as.Date(sapply(1:ndays,congress_getDate),origin="1970-01-01"),
                         as.numeric(sapply(1:ndays,congress_getApproval)))
names(congress_df) = c("date","app")

# Get rid of duplicates
congress_df_shortened = congress_df[cumsum(rle(as.numeric(congress_df[,2]))$lengths),]
x = as.integer(congress_df_shortened$date)
y = congress_df_shortened$app

# Plot data
plot(as.Date(x,origin="1970-01-01"),y,type="p",pch=4,cex=.5,ylab = "Percent Approve",
     xlab = "Time",main = "RCP Average Congress Approval Ratings")

# Define rational quadratic kernel
# Note the postive parameters are parameterized by their log
n_params = 5
rq = function(params,i,j) {
  h = exp(params[1])
  alpha = exp(params[2])
  l = exp(params[3])
  return(h^2 * (1 + (i-j)^2/(2*alpha*l^2))^(-alpha))
}

# Create the covariance matrix
cov_matrix = function(params,x1,x2) {
  K = matrix(0, nrow = length(x1), ncol = length(x2))
  for (i in 1:length(x1)) {
    for (j in 1:length(x2)) {
      K[i,j] = rq(params,x1[i],x2[j])
    }
  }
  return(K)
}

# Loss function
neg_log_marginal_likelihood = function(params,x,y){
  noise_scale = exp(params[4])
  mean_param = params[5]
  K_xx = cov_matrix(params,x,x) + noise_scale*diag(length(x))
  return(-1*dmvnorm(y,mean = rep(mean_param,length(y)), sigma = K_xx,log=TRUE))
}

# Make predictions
predictive_mean_and_cov = function(params,x,y,x_star) {
  noise_scale = exp(params[4])
  mean_param = params[5]
  K_xx = cov_matrix(params,x,x) + noise_scale*diag(length(x))
  K_xx_star = cov_matrix(params,x,x_star)
  K_x_star_x_star = cov_matrix(params,x_star,x_star)
  pred_mean = mean_param + t(K_xx_star) %*% solve(K_xx) %*% (y - mean_param)
  pred_cov = K_x_star_x_star - t(K_xx_star) %*% solve(K_xx) %*% K_xx_star
  return(list(pred_mean,pred_cov))
}

# Make plot with predictive mean and 95% confidence interval
plot_predictions = function(delta,params,x,y) {
  x_star = (min(x)):(max(x)+delta)
  predictions = predictive_mean_and_cov(params,x,y,x_star)
  pred_mean = predictions[[1]]
  pred_cov = predictions[[2]]
  
  marg_std = sqrt(diag(pred_cov))
  lower_bound = pred_mean - 1.96*marg_std
  upper_bound = pred_mean + 1.96*marg_std
  
  x_date = as.Date(x,origin="1970-01-01")
  plot(x_date,y,type="p",pch=4,cex=.5,xlim = c(min(x),max(x)+delta),ylim = c(min(y)*.97,max(y)*1.03),
       ylab = "Percent Approve",xlab = "Time",main = "RCP Average Congress Approval Ratings")
  polygon(c(rev(x_star), x_star), c(rev(upper_bound), lower_bound), col=adjustcolor("violet",alpha.f=0.5), 
          border = NA,xlim =c(-4,2))
  lines(x_star,pred_mean,lty='dashed',col= 'red')
}

# Use optim function to get hyper-parameters
optimized_params = optim(rnorm(n_params), neg_log_marginal_likelihood, x=x,y=y,method = 'CG')
params = optimized_params$par

# Graph next 100 days
days_ahead = 100
plot_predictions(days_ahead,params,x,y)

# Get predictions for January 9
target_date = as.integer(as.Date("2017-01-09"))
predictions = predictive_mean_and_cov(params,x,y,c(target_date))
mu = predictions[[1]]
stdev = sqrt(predictions[[2]])

# Calculate the probabilities of each PredictIt bucket
t1 = 1 - pnorm(14.95,mean=mu,sd= stdev)
t2 = pnorm(14.95,mean=mu,sd=stdev)-pnorm(14.45,mean=mu,sd=stdev)
t3 = pnorm(14.45,mean=mu,sd=stdev)-pnorm(13.95,mean=mu,sd=stdev)
t4 = pnorm(13.95,mean=mu,sd=stdev)-pnorm(13.45,mean=mu,sd=stdev)
t5 = pnorm(13.45,mean=mu,sd=stdev)

# Display rounded predictions
preds = c(t1,t2,t3,t4,t5)
round(preds,2)
