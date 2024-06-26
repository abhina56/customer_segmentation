import KhaltiCheckout from 'khalti-checkout-web';
import Button from '@mui/joy/Button';
import axios from 'axios';

const PaymentButton = () => {
  const handleClick = () => {
    var config = {
      publicKey: 'test_public_key_228666a712964f4aa3c13948c171',
      productIdentity: '1234567890',
      productName: 'Bus-Ticket',
      productUrl: 'http://localhost:4000/payment',
      paymentPreference: ['KHALTI', 'EBANKING', 'MOBILE_BANKING', 'CONNECT_IPS', 'SCT'],
      eventHandler: {
        onSuccess(payload) {
          console.log(payload);
          axios.post('http://localhost:5000/api/khalti', { token: payload.token, amount: 1000 })
            .then(response => {
              console.log(response.data);
            })
            .catch(error => {
              console.error(error);
            });
        },
        onError(error) {
          console.log(error);
        },
        onClose() {
          console.log('Widget is closing');
        }
      }
    };

    var checkout = new KhaltiCheckout(config);
    checkout.show({ amount: 1000 });
  };
};
export default PaymentButton;