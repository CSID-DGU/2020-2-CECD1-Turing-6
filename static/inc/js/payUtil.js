var APPID = "5c35df57b6d49c67f6bf7023";

function requestPayment(orderId, unique, price, name, userName, userEmail, userAddr, userPhone){
    BootPay.request({
        price: price,
        application_id: APPID,
        name: name,
        // pg: 'danal',
        // method: 'phone', //결제수단, 입력하지 않으면 결제수단 선택부터 화면이 시작합니다.
        show_agree_window: 0, // 부트페이 정보 동의 창 보이기 여부
        items: [
            {
                item_name: name, //상품명
                qty: 1, //수량
                unique: unique, // 단순 포인트 결제이므로 포인트 금액을 입력(원래는 구분용 Unique ID)
                price: price,
                cat1: 'POINT', // 대표 상품의 카테고리 상, 50글자 이내
                cat2: 'CHARGE', // 대표 상품의 카테고리 중, 50글자 이내
                cat3: 'ACTION', // 대표상품의 카테고리 하, 50글자 이내
            }
        ],
        user_info: {
            username: userName,
            email: userEmail,
            addr: userAddr,
            phone: userPhone
        },
        order_id: orderId, //고유 주문번호로, 생성하신 값을 보내주셔야 합니다.
        params: {payUnique: orderId, payUser: userName, price: price, unique: unique},
        // account_expire_at: '2018-05-25', // 가상계좌 입금기간 제한 ( yyyy-mm-dd 포멧으로 입력해주세요. 가상계좌만 적용됩니다. )
        extra: {
            // start_at: '2018-10-10', // 정기 결제 시작일 - 시작일을 지정하지 않으면 그 날 당일로부터 결제가 가능한 Billing key 지급
            // end_at: '2021-10-10', // 정기결제 만료일 -  기간 없음 - 무제한
            // vbank_result: 1, // 가상계좌 사용시 사용, 가상계좌 결과창을 볼지(1), 말지(0), 미설정시 봄(1)
            // quota: '0,2,3' // 결제금액이 5만원 이상시 할부개월 허용범위를 설정할 수 있음, [0(일시불), 2개월, 3개월] 허용, 미설정시 12개월까지 허용
        }
    }).error(function (data) {
        console.log(data);
        swal ( "알림" ,  "오류가 발생하였습니다.\n관리자에게 문의하세요.", "error" );
    }).cancel(function (data) {
        console.log(data);
        swal ( "알림" ,  "결제가 취소되었습니다.", "info" );
    }).ready(function (data) {
        // Virtual Bank
        console.log(data);
    }).confirm(function (data) {
        // Right Before the process - Stock Check needed / Not working on Card Manual Pay
        console.log(data);
        if (true) { // 재고 수량 관리 로직 혹은 다른 처리
            this.transactionConfirm(data); // 조건이 맞으면 승인 처리를 한다.
        } else {
            this.removePaymentWindow(); // 조건이 맞지 않으면 결제 창을 닫고 결제를 승인하지 않는다.
        }
    }).close(function (data) {
        // When all of closing Situation
        console.log(data);
    }).done(function (data) {
        // Done - Validation needed
        data.originPrice = price;
        data.originCharge = unique;
        console.log(data);
        callJson(
            "/csweb/shared/public/route.php?F=PayRoute.validatePayment",
            {
                data : data
            },
            function(data){
                if(data.returnCode == 1){
                    location.reload();
                }else{
                    swal ( "알림" ,  data.returnMessage, "error" );
                }
        });

    });
}
