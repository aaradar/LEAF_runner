#!/bin/bash

# Set environment variables (in a secure way, not hardcoded in the script in real scenarios)
export DOMAIN=csadev
export USERNAME=leaf
export PASSWORD=changeme
export CLIENT_ID=ws-leaf
export CLIENT_SECRET=changeme
export WORKSPACE=leaf


# Issue the curl command
curl -k --location "https://keycloak.${DOMAIN}.dec.earthdaily.com/realms/master/protocol/openid-connect/token" \
--header "Content-Type: application/x-www-form-urlencoded" \
--data-urlencode "grant_type=password" \
--data-urlencode "username=${USERNAME}" \
--data-urlencode "password=${PASSWORD}" \
--data-urlencode "client_secret=${CLIENT_SECRET}" \
--data-urlencode "client_id=${CLIENT_ID}"




'''
The correct response of executing above script are as follows:

 % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   464  100   464    0     0    738      0 --:--:-- --:--:-- --:--:--   742{"status":"ready","endpoints":[{"id":"resource-protection","url":"resource-catalogue.ws-leaf.csadev.dec.earthdaily.com"}],"storage":{"credentials":{"access":"user-BgceX1SSlUwerA","bucketname":"ws-leaf","projectid":"","secret":"Hr9rPk5A-G9795thY8eKMg","endpoint":"https://minio.csadev.dec.earthdaily.com","region":"RegionOne"}},"container_registry":{"username":"ws-leaf","password":"pKq645XhHLdeQ3pwA5V4wA4iUy1RqO","url":"https://harbor.csadev.dec.earthdaily.com"}}
'''