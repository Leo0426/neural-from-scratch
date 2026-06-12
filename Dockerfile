FROM nginx:1.27-alpine

COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY visualizations/ /usr/share/nginx/html/

EXPOSE 80

ENTRYPOINT []
CMD ["nginx", "-g", "daemon off;"]
